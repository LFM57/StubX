#!/usr/bin/env python3
# compiler.py
"""
Stubx -> Python transpiler + runner (Phase 3 - V3).
Usage:
    python compiler.py --file script.stubx         # compile and run
    python compiler.py --file script.stubx --no-run # just compile
    python compiler.py --repl                      # start interactive shell
    python compiler.py --format script.stubx       # pretty-print source
"""

from dataclasses import dataclass
import re
import sys
import argparse
import traceback
from typing import List, Optional, Dict, Any, Tuple
import os

# --------------------------- 
# Lexer
# --------------------------- 

TokenSpec = [
    ('COMMENT',  r'--\[(?s:.*?)(\s*)?\]--|--[^\n]*'),
    ('LET',      r'\blet\b'),
    ('NUMBER',   r'\d+(\.\d+)?n?'),
    ('STRING',   r'"([^"\\]|\\.)*"[ns]?'),
    ('BOOL',     r'(true|false)\??'),
    ('ASK',      r'\bask\b'),
    ('SAY',      r'\bsay\b'),
    ('FN',       r'\bfn\b'),
    ('RETURN',   r'\breturn\b'),
    ('IF',       r'\bif\b'),
    ('ELSE',     r'\belse\b'),
    ('WHILE',    r'\bwhile\b'),
    ('FOR',      r'\bfor\b'),
    ('IN',       r'\bin\b'),
    ('ATTEMPT',  r'\battempt\b'),
    ('RECOVER',  r'\brecover\b'),
    ('END',      r'\bend\b'),
    ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'),
    ('PIPE',     r'\|>'),
    ('DOTDOT',   r'\.\.'),
    ('ARROW',    r'->'),
    ('TILDE',    r'~'),
    ('OP',       r'==|!=|<=|>=|&&|\|\|[+\-*/%<>]'),
    ('LPAREN',   r'\('), 
    ('RPAREN',   r'\)'),
    ('LBRACKET', r'\['),
    ('RBRACKET', r'\]'),
    ('COLON',    r':'),
    ('COMMA',    r','),
    ('ASSIGN',   r'='),
    ('NEWLINE',  r'\n'),
    ('SKIP',     r'[ \t\r]+'),
]

MASTER_RE = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in TokenSpec), re.MULTILINE)

@dataclass
class Token:
    type: str
    value: str
    line: int
    col: int
    def __repr__(self):
        return f"Token({self.type},{self.value!r},ln={self.line},col={self.col})"

def lex(code: str) -> List[Token]:
    pos = 0
    line = 1
    col = 1
    tokens: List[Token] = []
    while pos < len(code):
        m = MASTER_RE.match(code, pos)
        if not m:
            snippet = code[pos:pos+20].replace('\n','\\n')
            raise SyntaxError(f"Unexpected character at line {line}, col {col}: {snippet!r}")
        typ = m.lastgroup
        val = m.group(typ)
        if typ == 'NEWLINE':
            tokens.append(Token('NEWLINE', val, line, col))
            pos = m.end()
            line += 1
            col = 1
            continue
        if typ == 'SKIP' or typ == 'COMMENT':
            newlines = val.count('\n')
            if newlines:
                line += newlines
                col = len(val) - val.rfind('\n')
            else:
                col += len(val)
            pos = m.end()
            continue
        tok = Token(typ, val, line, col)
        tokens.append(tok)
        pos = m.end()
        newlines = val.count('\n')
        if newlines:
            line += newlines
            col = len(val) - val.rfind('\n')
        else:
            col += len(val)
    tokens.append(Token('EOF', '', line, col))
    return tokens

# --------------------------- 
# AST nodes
# --------------------------- 

class ASTNode:
    pass

@dataclass
class Program(ASTNode):
    body: List[ASTNode]

@dataclass
class Assign(ASTNode):
    name: str
    expr: ASTNode
    line: int
    col: int

@dataclass
class Ask(ASTNode):
    name: str
    line: int
    col: int

@dataclass
class Fn(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]
    line: int
    col: int

@dataclass
class Return(ASTNode):
    expr: ASTNode
    line: int
    col: int

@dataclass
class If(ASTNode):
    cond: ASTNode
    then_block: List[ASTNode]
    else_block: Optional[List[ASTNode]]
    line: int
    col: int

@dataclass
class While(ASTNode):
    cond: ASTNode
    body: List[ASTNode]
    line: int
    col: int

@dataclass
class For(ASTNode):
    var_name: str
    iterable: ASTNode
    body: List[ASTNode]
    line: int
    col: int

@dataclass
class TryCatch(ASTNode):
    try_block: List[ASTNode]
    recover_block: List[ASTNode]
    line: int
    col: int

@dataclass
class Say(ASTNode):
    args: List[ASTNode]
    line: int
    col: int

@dataclass
class ExpressionStmt(ASTNode):
    expr: ASTNode
    line: int
    col: int

@dataclass
class ListLit(ASTNode):
    items: List[ASTNode]
    line: int
    col: int

@dataclass
class Range(ASTNode):
    start: ASTNode
    end: ASTNode
    line: int
    col: int

@dataclass
class Call(ASTNode):
    name: str
    args: List[ASTNode]
    line: int
    col: int

# expressions
@dataclass
class Number(ASTNode):
    value: str
    line: int
    col: int

@dataclass
class String(ASTNode):
    value: str
    line: int
    col: int

@dataclass
class Bool(ASTNode):
    value: str
    line: int
    col: int

@dataclass
class ImplicitVar(ASTNode):
    line: int
    col: int

@dataclass
class Var(ASTNode):
    name: str
    line: int
    col: int

@dataclass
class BinOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode
    line: int
    col: int

# --------------------------- 
# Parser
# --------------------------- 

class ParseError(Exception):
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t.type != 'NEWLINE']
        self.i = 0

    def cur(self) -> Token:
        return self.tokens[self.i]

    def eat(self, typ: Optional[str] = None) -> Token:
        t = self.cur()
        if typ and t.type != typ:
            raise ParseError(f"Expected {typ} at line {t.line} col {t.col}, got {t.type} ({t.value})")
        self.i += 1
        return t

    def parse(self) -> Program:
        stmts = []
        while self.cur().type != 'EOF':
            stmts.append(self.parse_stmt())
        return Program(stmts)

    def parse_stmt(self) -> ASTNode:
        t = self.cur()
        
        if t.type == 'LET':
            lt = self.eat('LET')
            idt = self.eat('ID')
            self.eat('ASSIGN')
            expr = self.parse_expr()
            return Assign(idt.value, expr, lt.line, lt.col)
        
        if t.type == 'ASK':
            at = self.eat('ASK')
            idt = self.eat('ID')
            return Ask(idt.value, at.line, at.col)
        
        if t.type == 'SAY':
            st = self.eat('SAY')
            args = [self.parse_expr()]
            while self.cur().type == 'COMMA':
                self.eat('COMMA')
                args.append(self.parse_expr())
            return Say(args, st.line, st.col)

        if t.type == 'IF':
            return self.parse_if()
        
        if t.type == 'WHILE':
            wt = self.eat('WHILE')
            cond = self.parse_expr()
            body = self.parse_block()
            self.eat('END')
            return While(cond, body, wt.line, wt.col)
            
        if t.type == 'FOR':
            ft = self.eat('FOR')
            var_name = self.eat('ID').value
            self.eat('IN')
            iterable = self.parse_expr()
            body = self.parse_block()
            self.eat('END')
            return For(var_name, iterable, body, ft.line, ft.col)

        if t.type == 'ATTEMPT':
            return self.parse_try_catch()

        if t.type == 'RETURN':
            rt = self.eat('RETURN')
            expr = self.parse_expr()
            return Return(expr, rt.line, rt.col)

        if t.type == 'ID':
            peek1 = self.tokens[self.i+1] if self.i+1 < len(self.tokens) else Token('EOF','',0,0)
            
            if peek1.type == 'ASSIGN':
                 raise ParseError(f"Assignments must use 'let' keyword at line {t.line} col {t.col}")
            
            if peek1.type == 'LPAREN':
                saved_i = self.i
                try:
                    self.eat('ID')
                    self.eat('LPAREN')
                    depth = 1
                    while depth > 0 and self.i < len(self.tokens):
                        tt = self.cur().type
                        if tt == 'LPAREN': depth += 1
                        elif tt == 'RPAREN': depth -= 1
                        self.i += 1
                    
                    if depth == 0 and self.i < len(self.tokens):
                        after_paren = self.cur()
                        if after_paren.type in ('ARROW', 'COLON'):
                            self.i = saved_i
                            idt = self.eat('ID')
                            self.eat('LPAREN')
                            params = []
                            if self.cur().type != 'RPAREN':
                                params.append(self.eat('ID').value)
                                while self.cur().type == 'COMMA':
                                    self.eat('COMMA')
                                    params.append(self.eat('ID').value)
                            self.eat('RPAREN')
                            
                            if self.cur().type == 'ARROW':
                                self.eat('ARROW')
                                body_expr = self.parse_expr()
                                return Fn(idt.value, params, [Return(body_expr, body_expr.line, body_expr.col)], idt.line, idt.col)
                            else:
                                self.eat('COLON')
                                stmts = []
                                while self.cur().type not in ('END', 'ELSE', 'EOF', 'RECOVER'):
                                    stmts.append(self.parse_stmt())
                                self.eat('END')
                                return Fn(idt.value, params, stmts, idt.line, idt.col)
                except Exception:
                    pass
                self.i = saved_i

        start_node = self.parse_expr()
        return ExpressionStmt(start_node, start_node.line, start_node.col)

    def parse_if(self) -> If:
        it = self.eat('IF')
        cond = self.parse_expr()
        then_block = self.parse_block()
        else_block = None
        if self.cur().type == 'ELSE':
            self.eat('ELSE')
            else_block = self.parse_block()
        self.eat('END')
        return If(cond, then_block, else_block, it.line, it.col)

    def parse_try_catch(self) -> TryCatch:
        at = self.eat('ATTEMPT')
        try_block = self.parse_block()
        self.eat('RECOVER')
        recover_block = self.parse_block()
        self.eat('END')
        return TryCatch(try_block, recover_block, at.line, at.col)

    def parse_block(self) -> List[ASTNode]:
        self.eat('COLON')
        stmts = []
        while self.cur().type not in ('END', 'ELSE', 'RECOVER', 'EOF'):
            stmts.append(self.parse_stmt())
        return stmts

    def parse_expr(self) -> ASTNode:
        return self.parse_binop()

    def parse_juxtaposition(self) -> ASTNode:
        left = self.parse_primary()
        while self.cur().type in ('ID', 'NUMBER', 'STRING', 'BOOL', 'TILDE', 'LBRACKET', 'LPAREN'):
             if self.cur().line > left.line:
                 break
             arg = self.parse_primary()
             if isinstance(left, Call):
                  left.args.append(arg)
             elif isinstance(left, Var):
                  left = Call(left.name, [arg], left.line, left.col)
             else:
                  raise ParseError(f"Invalid function call target at line {left.line}")
        return left

    def parse_primary(self) -> ASTNode:
        t = self.cur()
        if t.type == 'NUMBER':
            tok = self.eat('NUMBER')
            val = tok.value[:-1] if tok.value.endswith('n') else tok.value
            return Number(val, tok.line, tok.col)
        if t.type == 'STRING':
            tok = self.eat('STRING')
            raw = tok.value
            if raw.endswith('n'):
                val = raw[1:-2]
                return Number(val, tok.line, tok.col)
            elif raw.endswith('s'):
                val = raw[:-1]
                return String(val, tok.line, tok.col)
            else:
                val = raw
                return String(val, tok.line, tok.col)
        if t.type == 'BOOL':
            tok = self.eat('BOOL')
            raw = tok.value[:-1] if tok.value.endswith('?') else tok.value
            val = "True" if raw == "true" else "False"
            return Bool(val, tok.line, tok.col)
        if t.type == 'TILDE':
            tok = self.eat('TILDE')
            return ImplicitVar(tok.line, tok.col)
        if t.type == 'LBRACKET':
            lt = self.eat('LBRACKET')
            items = []
            if self.cur().type != 'RBRACKET':
                items.append(self.parse_expr())
                while self.cur().type == 'COMMA':
                    self.eat('COMMA')
                    items.append(self.parse_expr())
            self.eat('RBRACKET')
            return ListLit(items, lt.line, lt.col)
        if t.type == 'ID':
            tok = self.eat('ID')
            if self.cur().type == 'LPAREN':
                self.eat('LPAREN')
                args = []
                if self.cur().type != 'RPAREN':
                    args.append(self.parse_expr())
                    while self.cur().type == 'COMMA':
                        self.eat('COMMA')
                        args.append(self.parse_expr())
                self.eat('RPAREN')
                return Call(tok.value, args, tok.line, tok.col)
            return Var(tok.value, tok.line, tok.col)
        if t.type == 'LPAREN':
            self.eat('LPAREN')
            e = self.parse_expr()
            self.eat('RPAREN')
            return e
        if t.type == 'SAY':
            self.eat('SAY')
            return Var('print', t.line, t.col)
        if t.type == 'ASK':
            self.eat('ASK')
            return Var('input', t.line, t.col)
            
        raise ParseError(f"Unexpected in primary: {t.type} at line {t.line} col {t.col}")

    def parse_binop(self, min_prec=0) -> ASTNode:
        left = self.parse_juxtaposition()
        prec = {
            '|>': 0,
            '||': 1, '&&': 2,
            '==':3, '!=':3, '<':4, '>':4, '<=':4, '>=':4,
            '+':5, '-':5,
            '*':6, '/':6, '%':6,
            '..': 7
        }
        while self.cur().type in ('OP', 'PIPE', 'DOTDOT') and prec.get(self.cur().value, 0) >= min_prec:
            op_tok = self.cur()
            self.eat(op_tok.type)
            op = op_tok.value
            p = prec.get(op, 0)
            right = self.parse_binop(p+1)
            
            if op == '|>':
                if isinstance(right, Call):
                    right.args.insert(0, left)
                    left = right
                elif isinstance(right, Var):
                    left = Call(right.name, [left], op_tok.line, op_tok.col)
                else:
                    raise ParseError(f"Invalid pipe target at line {op_tok.line}")
            elif op == '..':
                left = Range(left, right, op_tok.line, op_tok.col)
            else:
                left = BinOp(op, left, right, op_tok.line, op_tok.col)
        return left

# --------------------------- 
# Formatter (V3)
# --------------------------- 

class Formatter:
    def __init__(self):
        self.indent = 0
    
    def i(self):
        return "    " * self.indent
        
    def format(self, node: ASTNode) -> str:
        if isinstance(node, Program):
            return "\n".join(self.format(s) for s in node.body)
        elif isinstance(node, Assign):
            return f"{self.i()}let {node.name} = {self.fmt_expr(node.expr)}"
        elif isinstance(node, Ask):
            return f"{self.i()}ask {node.name}"
        elif isinstance(node, Say):
            args = ", ".join(self.fmt_expr(a) for a in node.args)
            return f"{self.i()}say {args}"
        elif isinstance(node, Return):
            return f"{self.i()}return {self.fmt_expr(node.expr)}"
        elif isinstance(node, ExpressionStmt):
            return f"{self.i()}{self.fmt_expr(node.expr)}"
        elif isinstance(node, If):
            out = f"{self.i()}if {self.fmt_expr(node.cond)}:\n"
            self.indent += 1
            for s in node.then_block: out += self.format(s) + "\n"
            self.indent -= 1
            if node.else_block is not None:
                out += f"{self.i()}else:\n"
                self.indent += 1
                for s in node.else_block: out += self.format(s) + "\n"
                self.indent -= 1
            out += f"{self.i()}end"
            return out
        elif isinstance(node, While):
            out = f"{self.i()}while {self.fmt_expr(node.cond)}:\n"
            self.indent += 1
            for s in node.body: out += self.format(s) + "\n"
            self.indent -= 1
            out += f"{self.i()}end"
            return out
        elif isinstance(node, For):
            out = f"{self.i()}for {node.var_name} in {self.fmt_expr(node.iterable)}:\n"
            self.indent += 1
            for s in node.body: out += self.format(s) + "\n"
            self.indent -= 1
            out += f"{self.i()}end"
            return out
        elif isinstance(node, TryCatch):
            out = f"{self.i()}attempt:\n"
            self.indent += 1
            for s in node.try_block: out += self.format(s) + "\n"
            self.indent -= 1
            out += f"{self.i()}recover:\n"
            self.indent += 1
            for s in node.recover_block: out += self.format(s) + "\n"
            self.indent -= 1
            out += f"{self.i()}end"
            return out
        elif isinstance(node, Fn):
            out = f"{self.i()}fn {node.name}({', '.join(node.params)}):\n"
            self.indent += 1
            for s in node.body: out += self.format(s) + "\n"
            self.indent -= 1
            out += f"{self.i()}end"
            return out
        return f"{self.i()}-- Unknown Node {type(node)}"

    def fmt_expr(self, expr: ASTNode) -> str:
        if isinstance(expr, Number): return expr.value
        if isinstance(expr, String): return f'\"{expr.value}"' # Corrected escaping for string literal
        if isinstance(expr, Bool): return expr.value.lower()
        if isinstance(expr, ImplicitVar): return "~"
        if isinstance(expr, Var): return expr.name
        if isinstance(expr, ListLit): return f"[{', '.join(self.fmt_expr(x) for x in expr.items)}]"
        if isinstance(expr, Range): return f"{self.fmt_expr(expr.start)}..{self.fmt_expr(expr.end)}"
        if isinstance(expr, BinOp):
            return f"{self.fmt_expr(expr.left)} {expr.op} {self.fmt_expr(expr.right)}"
        if isinstance(expr, Call):
            return f"{expr.name}({', '.join(self.fmt_expr(a) for a in expr.args)})"
        return ""

# --------------------------- 
# Code generator
# --------------------------- 

class CodeGen:
    def __init__(self, include_header=True, repl_mode=False):
        self.lines: List[str] = []
        self.indent = 0
        self.line_map: Dict[int, Tuple[int,int]] = {}
        self.include_header = include_header
        self.repl_mode = repl_mode

    def writeln(self, s: str = '', src_pos: Optional[Tuple[int,int]] = None):
        py_line_no = len(self.lines) + 1
        indent_str = '    ' * self.indent
        self.lines.append(indent_str + s)
        if src_pos:
            self.line_map[py_line_no] = src_pos

    def gen(self, node: ASTNode):
        if isinstance(node, Program):
            if self.include_header:
                self.writeln("import random")
                self.writeln("import subprocess")
                self.writeln("import os")
                
                self.writeln("def _stubx_add(a, b):")
                self.writeln("    if isinstance(a, str) or isinstance(b, str):")
                self.writeln("        return str(a) + str(b)")
                self.writeln("    return a + b")
                self.writeln("def _stubx_range(a, b): return range(a, b + 1)")
                self.writeln("def _stubx_random(x):")
                self.writeln("    if isinstance(x, int): return random.randint(0, x)")
                self.writeln("    return random.choice(x)")
                self.writeln("def _stubx_upper(x): return str(x).upper()")
                
                self.writeln("def _stubx_read(path):")
                self.writeln("    with open(str(path), 'r', encoding='utf-8') as f: return f.read()")
                self.writeln("def _stubx_append(content, path):")
                self.writeln("    with open(str(path), 'a', encoding='utf-8') as f: f.write(str(content))")
                self.writeln("def _stubx_exec(cmd):")
                self.writeln("    return subprocess.check_output(str(cmd), shell=True, text=True).strip()")
                self.writeln("def _stubx_map(func, iterable): return list(map(func, iterable))")
                self.writeln("def _stubx_filter(func, iterable): return list(filter(func, iterable))")

                self.writeln("")
                self.writeln("_last_val = None")
            
            for n in node.body:
                self.gen(n)
        elif isinstance(node, Assign):
            expr_str = self.gen_expr(node.expr)
            self.writeln(f"{node.name} = {expr_str}", (node.line, node.col))
            self.writeln(f"_last_val = {node.name}")
            if self.repl_mode: self.writeln("_repl_has_val = True")
        elif isinstance(node, Ask):
            self.writeln(f"{node.name} = input()", (node.line, node.col))
        elif isinstance(node, Fn):
            self.writeln(f"def {node.name}({', '.join(node.params)}):", (node.line, node.col))
            self.indent += 1
            if not node.body:
                self.writeln("pass")
            else:
                for s in node.body:
                    self.gen(s)
            self.indent -= 1
            self.writeln("")
        elif isinstance(node, Return):
            self.writeln("return " + self.gen_expr(node.expr), (node.line, node.col))
        elif isinstance(node, If):
            self.writeln(f"if {self.gen_expr(node.cond)}:", (node.line, node.col))
            self.indent += 1
            if not node.then_block:
                self.writeln("pass")
            else:
                for s in node.then_block:
                    self.gen(s)
            self.indent -= 1
            if node.else_block is not None:
                self.writeln("else:", (node.line, node.col))
                self.indent += 1
                if not node.else_block:
                    self.writeln("pass")
                else:
                    for s in node.else_block:
                        self.gen(s)
                self.indent -= 1
        elif isinstance(node, TryCatch):
            self.writeln("try:", (node.line, node.col))
            self.indent += 1
            if not node.try_block:
                self.writeln("pass")
            else:
                for s in node.try_block:
                    self.gen(s)
            self.indent -= 1
            self.writeln("except Exception:", (node.line, node.col))
            self.indent += 1
            if not node.recover_block:
                self.writeln("pass")
            else:
                for s in node.recover_block:
                    self.gen(s)
            self.indent -= 1
        elif isinstance(node, While):
            self.writeln(f"while {self.gen_expr(node.cond)}:", (node.line, node.col))
            self.indent += 1
            if not node.body:
                self.writeln("pass")
            else:
                for s in node.body:
                    self.gen(s)
            self.indent -= 1
        elif isinstance(node, For):
            self.writeln(f"for {node.var_name} in {self.gen_expr(node.iterable)}:", (node.line, node.col))
            self.indent += 1
            if not node.body:
                self.writeln("pass")
            else:
                for s in node.body:
                    self.gen(s)
            self.indent -= 1
        elif isinstance(node, Say):
            args = ", ".join(self.gen_expr(a) for a in node.args)
            self.writeln(f"print({args})", (node.line, node.col))
        elif isinstance(node, Call):
            self.writeln(f"{self.gen_call(node)}", (node.line, node.col))
        elif isinstance(node, ExpressionStmt):
            self.writeln(f"_last_val = {self.gen_expr(node.expr)}", (node.line, node.col))
            if self.repl_mode: self.writeln("_repl_has_val = True")
        else:
            raise NotImplementedError(f"CodeGen: unhandled node type {type(node)}")

    def gen_call(self, node: Call) -> str:
        if node.name == 'len':
            return f"len({', '.join(self.gen_expr(a) for a in node.args)})"
        if node.name == 'random':
            return f"_stubx_random({', '.join(self.gen_expr(a) for a in node.args)})"
        if node.name == 'upper':
            return f"_stubx_upper({', '.join(self.gen_expr(a) for a in node.args)})"
        if node.name == 'int':
            return f"int({', '.join(self.gen_expr(a) for a in node.args)})"
        if node.name == 'str':
            return f"str({', '.join(self.gen_expr(a) for a in node.args)})"
        if node.name == 'read':
            return f"_stubx_read({', '.join(self.gen_expr(a) for a in node.args)})"
        if node.name == 'append_to':
            return f"_stubx_append({', '.join(self.gen_expr(a) for a in node.args)})"
        if node.name == 'exec':
            return f"_stubx_exec({', '.join(self.gen_expr(a) for a in node.args)})"
        if node.name == 'map':
            return f"_stubx_map({', '.join(self.gen_expr(a) for a in node.args)})"
        if node.name == 'filter':
            return f"_stubx_filter({', '.join(self.gen_expr(a) for a in node.args)})"
            
        return f"{node.name}({', '.join(self.gen_expr(a) for a in node.args)})"

    def gen_expr(self, expr: ASTNode) -> str:
        if isinstance(expr, Number):
            return expr.value
        if isinstance(expr, String):
            return expr.value # String literals are already correctly escaped
        if isinstance(expr, Bool):
            return expr.value
        if isinstance(expr, ImplicitVar):
            return "_last_val"
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, ListLit):
            return f"[{', '.join(self.gen_expr(x) for x in expr.items)} ]"
        if isinstance(expr, Range):
            return f"_stubx_range({self.gen_expr(expr.start)}, {self.gen_expr(expr.end)})"
        if isinstance(expr, BinOp):
            left = self.gen_expr(expr.left)
            right = self.gen_expr(expr.right)
            if expr.op == '+':
                return f"_stubx_add({left}, {right})"
            return f"({left} {expr.op} {right})"
        if isinstance(expr, Call):
            return self.gen_call(expr)
        raise NotImplementedError(f"gen_expr: {type(expr)}")

    def result(self) -> str:
        return "\n".join(self.lines)

# --------------------------- 
# Transpile + run utilities
# --------------------------- 

def transpile(code: str) -> Tuple[str, Dict[int, Tuple[int,int]]]:
    tokens = lex(code)
    parser = Parser(tokens)
    try:
        ast = parser.parse()
    except ParseError as e:
        raise ParseError(f"Parse error: {e}")
    cg = CodeGen()
    cg.gen(ast)
    py = cg.result()
    return py, cg.line_map

def write_py(py_code: str, src_path: str) -> str:
    out_path = src_path.rsplit('.', 1)[0] + '.py'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# Generated from " + src_path + "\n")
        f.write(py_code)
        f.write("\n")
    return out_path

def run_generated(py_code: str, line_map: Dict[int, Tuple[int,int]], src_path: str):
    globals_dict: Dict[str, Any] = {}
    try:
        compiled = compile(py_code, filename="<stubx_generated>", mode='exec')
        exec(compiled, globals_dict)
    except Exception as e:
        tb = e.__traceback__
        relevant = None
        while tb is not None:
            frame = tb.tb_frame
            lineno = tb.tb_lineno
            co = frame.f_code
            fname = co.co_filename
            if fname == "<stubx_generated>":
                relevant = (lineno, tb)
                break
            tb = tb.tb_next
        print("=== Exception during execution ===")
        if relevant:
            gen_lineno = relevant[0]
            src_pos = line_map.get(gen_lineno)
            if src_pos:
                print(f"Error: {e.__class__.__name__}: {e}")
                print(f"At Stubx source: {src_path} line {src_pos[0]} col {src_pos[1]} (mapped from generated line {gen_lineno})")
                try:
                    with open(src_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    src_line = lines[src_pos[0]-1].rstrip('\n')
                    print("    " + src_line)
                except Exception:
                    pass
            else:
                print("Could not map generated line to source. Traceback follows:")
                traceback.print_exception(e)
        else:
            print("No generated-file frame found in traceback; full traceback:")
            traceback.print_exception(e)

# --------------------------- 
# REPL
# --------------------------- 

def repl():
    print("StubX V3 REPL. Type 'exit' or 'quit' to leave.")
    
    # Initialize context with helpers
    cg = CodeGen(include_header=True, repl_mode=True)
    # Generate header code
    cg.gen(Program([])) 
    header_code = cg.result()
    
    ctx = {}
    try:
        exec(header_code, ctx)
    except Exception as e:
        print(f"REPL Init Error: {e}")
        return
        
    line_num = 1
    while True:
        try:
            text = input(f"stubx [{line_num}]> ")
            if text.strip() in ('exit', 'quit'):
                break
            if not text.strip():
                continue
                
            # Parse & Run
            try:
                tokens = lex(text)
                parser = Parser(tokens)
                ast = parser.parse()
                
                # Generate code for this snippet (no header)
                # Reset the flag before running
                ctx['_repl_has_val'] = False
                
                cg_snip = CodeGen(include_header=False, repl_mode=True)
                cg_snip.gen(ast)
                py_code = cg_snip.result()
                
                # Execute in persistent context
                exec(py_code, ctx)
                
                if ctx.get('_repl_has_val') and '_last_val' in ctx and ctx['_last_val'] is not None:
                    print(f"=> {ctx['_last_val']}")
                    
            except ParseError as e:
                print(f"Syntax Error: {e}")
            except SyntaxError as e:
                print(f"Lexer Error: {e}")
            except Exception as e:
                print(f"Runtime Error: {e}")
            
            line_num += 1
            
        except KeyboardInterrupt:
            print("\n(Use 'exit' to quit)")
        except EOFError:
            break

# --------------------------- 
# CLI
# --------------------------- 

def main(argv=None):
    p = argparse.ArgumentParser(description="Stubx transpiler (Phase 3 - V3).")
    p.add_argument('--file', '-f', help='Stubx source file (.stubx)')
    p.add_argument('--no-run', action='store_true', help='Only compile to .py, do not run')
    p.add_argument('--compile', action='store_true', help='Only execute the script, do not compile to code to a .py file')
    p.add_argument('--repl', action='store_true', help='Start REPL session')
    p.add_argument('--format', help='Format the given Stubx file and print to stdout')
    
    args = p.parse_args(argv)

    if args.repl:
        repl()
        return

    if args.format:
        src_path = args.format
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                src = f.read()
            tokens = lex(src)
            parser = Parser(tokens)
            ast = parser.parse()
            formatter = Formatter()
            print(formatter.format(ast))
        except Exception as e:
            print(f"Error formatting {src_path}: {e}")
        return

    if not args.file:
        p.print_help()
        return

    src_path = args.file
    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            src = f.read()
    except FileNotFoundError:
        print(f"File not found: {src_path}")
        sys.exit(2)

    try:
        py, line_map = transpile(src)
    except ParseError as e:
        print(f"Parse failure: {e}")
        sys.exit(1)
    except SyntaxError as e:
        print(f"Lexing/Syntax error: {e}")
        sys.exit(1)

    out_path = write_py(py, src_path)

    if args.compile or args.no_run:
        print("=== Generated Python (first 200 lines) ===")
        for i, line in enumerate(py.splitlines()[:200], start=1):
            mapped = line_map.get(i)
            m = f"  # <- Stubx ln {mapped[0]}" if mapped else ""
            print(f"{i:04d}: {line}{m}")

    if args.compile: 
        print(f"Compiled {src_path} -> {out_path}")

    if args.no_run:
        print(f"Compilation finished to {out_path}.")
        return

    print("=== Running... ===")
    run_generated(py, line_map, src_path)
    
    if not args.compile: 
        os.remove(out_path)


if __name__ == '__main__':
    main()
