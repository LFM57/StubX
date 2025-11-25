#!/usr/bin/env python3
# compilator.py
"""
Stubx -> Python transpiler + runner (Phase 1).
Usage:
    python compilator.py --file ./script.stubx        # compile and run
    python compilator.py --file ./script.stubx --no-run  # just compile to .py
"""

from dataclasses import dataclass
import re
import sys
import argparse
import traceback
from typing import List, Optional, Dict, Any, Tuple

# ---------------------------
# Lexer (tracks line/col)
# ---------------------------

TokenSpec = [
    ('COMMENT',  r'--\[(?s:.*?)\]--|--[^\n]*'),
    ('LET',      r'\blet\b'),
    ('NUMBER',   r'\d+(\.\d+)?n?'),
    ('STRING',   r'"([^"\\]|\\.)*"[ns]?'),
    ('BOOL',     r'(true|false)\??'),
    ('ASK',      r'\bask\b'),
    ('SAY',      r'\bsay\b'),
    ('FN',       r'\bfn\b'),  # Kept for backward compatibility if needed, but we'll prefer implicit
    ('RETURN',   r'\breturn\b'),
    ('IF',       r'\bif\b'),
    ('ELSE',     r'\belse\b'),
    ('WHILE',    r'\bwhile\b'),
    ('FOR',      r'\bfor\b'),
    ('IN',       r'\bin\b'),
    ('END',      r'\bend\b'),
    ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'),
    ('PIPE',     r'\|>'),
    ('DOTDOT',   r'\.\.'),
    ('ARROW',    r'->'),
    ('TILDE',    r'~'),
    ('OP',       r'==|!=|<=|>=|&&|\|\||[+\-*/%<>]'),
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
            # update line/col but do not emit token
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
        # update line/col for next token
        newlines = val.count('\n')
        if newlines:
            line += newlines
            col = len(val) - val.rfind('\n')
        else:
            col += len(val)
    tokens.append(Token('EOF', '', line, col))
    return tokens

# ---------------------------
# AST nodes (Phase 1)
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
# Parser (recursive descent) -> produces AST with position info
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
            # say expr (or multiple?) The example shows 'say "..." + name', which is one expr
            # But 'print' used to take comma args. Let's allow comma args for robustness or just one.
            # Spec says "say expr".
            args = [self.parse_expr()]
            # optional support for commas just in case user wants multiple
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

        if t.type == 'RETURN':
            rt = self.eat('RETURN')
            expr = self.parse_expr()
            return Return(expr, rt.line, rt.col)

        # ID can be:
        # 1. Assignment: x = ...
        # 2. Function Def: f(x) -> ... or f(x): ...
        # 3. Call (as statement): f(x) or f x
        # 4. Variable access (Expression Stmt): x
        if t.type == 'ID':
            # Look ahead
            peek1 = self.tokens[self.i+1] if self.i+1 < len(self.tokens) else Token('EOF','',0,0)
            
            if peek1.type == 'ASSIGN':
                # Assignment
                idt = self.eat('ID')
                self.eat('ASSIGN')
                expr = self.parse_expr()
                return Assign(idt.value, expr, idt.line, idt.col)
            
            if peek1.type == 'LPAREN':
                # Could be Function Def or Call or Juxtaposition start if parens are just grouping?
                # Actually parse_primary handles ID(...) as Call.
                # But we need to disambiguate Fn Def from Call.
                # Fn Def: ID ( args ) -> ... or ID ( args ) : ...
                # Call: ID ( args ) ...
                
                # We must parse the "ID ( ... )" part tentatively or check what follows RPAREN.
                # This requires a bit more lookahead or backtracking.
                # Let's try to parse the signature logic by manual peek/eat sequence,
                # essentially replicating parse_primary logic but pausing to check for ->/:
                
                # Save state
                saved_i = self.i
                try:
                    self.eat('ID')
                    self.eat('LPAREN')
                    # Scan until RPAREN
                    depth = 1
                    while depth > 0 and self.i < len(self.tokens):
                        tt = self.cur().type
                        if tt == 'LPAREN': depth += 1
                        elif tt == 'RPAREN': depth -= 1
                        self.i += 1
                    
                    if depth == 0 and self.i < len(self.tokens):
                        after_paren = self.cur()
                        if after_paren.type in ('ARROW', 'COLON'):
                            # It IS a Function Definition
                            self.i = saved_i # Restore
                            # Now parse properly
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
                                while self.cur().type not in ('END', 'ELSE', 'EOF'):
                                    stmts.append(self.parse_stmt())
                                self.eat('END') # Fn block end
                                return Fn(idt.value, params, stmts, idt.line, idt.col)
                except Exception:
                    pass
                self.i = saved_i # Restore if not Fn Def

        # Fallback to expression statement
        start_node = self.parse_expr()
        return ExpressionStmt(start_node, start_node.line, start_node.col)

    def parse_if(self) -> If:
        it = self.eat('IF')
        cond = self.parse_expr()
        # block expects ':'
        then_block = self.parse_block()
        else_block = None
        if self.cur().type == 'ELSE':
            self.eat('ELSE')
            else_block = self.parse_block()
        self.eat('END')
        return If(cond, then_block, else_block, it.line, it.col)

    def parse_block(self) -> List[ASTNode]:
        # Expects COLON, reads until END or ELSE
        self.eat('COLON')
        stmts = []
        while self.cur().type not in ('END', 'ELSE', 'EOF'):
            stmts.append(self.parse_stmt())
        return stmts

    # expressions (precedence climbing)
    def parse_expr(self) -> ASTNode:
        return self.parse_binop()

    def parse_juxtaposition(self) -> ASTNode:
        left = self.parse_primary()
        # Juxtaposition: valid starts for arguments
        # Debug print
        # print(f"DEBUG: Juxtaposition Left={left}, NextToken={self.cur().type} {self.cur().value}")
        while self.cur().type in ('ID', 'NUMBER', 'STRING', 'BOOL', 'TILDE', 'LBRACKET', 'LPAREN'):
             # Implicit statement boundary check: argument must be on same line as function
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
        # Allow keywords to be used as functions in expressions (Pipe target)
        if t.type == 'SAY':
            self.eat('SAY')
            return Var('print', t.line, t.col)
        if t.type == 'ASK':
            self.eat('ASK')
            return Var('input', t.line, t.col)
            
        raise ParseError(f"Unexpected in primary: {t.type} at line {t.line} col {t.col}")

    def parse_binop(self, min_prec=0) -> ASTNode:
        left = self.parse_juxtaposition() # Use juxtaposition here
        prec = {
            '|>': 0, # Pipe lowest
            '||': 1, '&&': 2,
            '==':3, '!=':3, '<':4, '>':4, '<=':4, '>=':4,
            '+':5, '-':5,
            '*':6, '/':6, '%':6,
            '..': 7 # Range high
        }
        # Accept OP, PIPE, DOTDOT
        while self.cur().type in ('OP', 'PIPE', 'DOTDOT') and prec.get(self.cur().value, 0) >= min_prec:
            op_tok = self.cur()
            self.eat(op_tok.type)
            op = op_tok.value
            p = prec.get(op, 0)
            
            # Right associative? Pipe is usually left. 
            # .. is usually non-associative or left.
            right = self.parse_binop(p+1)
            
            if op == '|>':
                # Transform Pipe to Call immediately? Or Keep as BinOp?
                # Let's keep as BinOp for AST simplicity, CodeGen handles it.
                # Actually, transforming now is easier for CodeGen.
                # a |> b -> b(a)
                if isinstance(right, Call):
                    # b(args) -> b(a, args)
                    right.args.insert(0, left)
                    left = right
                elif isinstance(right, Var):
                    # b -> b(a)
                    left = Call(right.name, [left], op_tok.line, op_tok.col)
                else:
                    raise ParseError(f"Invalid pipe target at line {op_tok.line}")
            elif op == '..':
                left = Range(left, right, op_tok.line, op_tok.col)
            else:
                left = BinOp(op, left, right, op_tok.line, op_tok.col)
        return left

# ---------------------------
# Code generator (AST -> Python), with line mapping
# ---------------------------

class CodeGen:
    def __init__(self):
        self.lines: List[str] = []
        self.indent = 0
        # mapping generated python line number (1-based) -> source stubx (line, col)
        self.line_map: Dict[int, Tuple[int,int]] = {}

    def writeln(self, s: str = '', src_pos: Optional[Tuple[int,int]] = None):
        py_line_no = len(self.lines) + 1
        indent_str = '    ' * self.indent
        self.lines.append(indent_str + s)
        if src_pos:
            self.line_map[py_line_no] = src_pos

    def gen(self, node: ASTNode):
        if isinstance(node, Program):
            self.writeln("import random")
            self.writeln("def _stubx_add(a, b):")
            self.writeln("    if isinstance(a, str) or isinstance(b, str):")
            self.writeln("        return str(a) + str(b)")
            self.writeln("    return a + b")
            self.writeln("def _stubx_range(a, b): return range(a, b + 1)")
            self.writeln("def _stubx_random(x):")
            self.writeln("    if isinstance(x, int): return random.randint(0, x)") # Handle random(10)
            self.writeln("    return random.choice(x)")
            self.writeln("def _stubx_upper(x): return str(x).upper()")
            self.writeln("")
            self.writeln("_last_val = None")
            for n in node.body:
                self.gen(n)
        elif isinstance(node, Assign):
            expr_str = self.gen_expr(node.expr)
            self.writeln(f"{node.name} = {expr_str}", (node.line, node.col))
            self.writeln(f"_last_val = {node.name}")
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
            self.writeln("if " + self.gen_expr(node.cond) + ":", (node.line, node.col))
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
        elif isinstance(node, While):
            self.writeln("while " + self.gen_expr(node.cond) + ":", (node.line, node.col))
            self.indent += 1
            if not node.body:
                self.writeln("pass")
            else:
                for s in node.body:
                    self.gen(s)
            self.indent -= 1
        elif isinstance(node, For):
            # for var in iterable:
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
        else:
            raise NotImplementedError(f"CodeGen: unhandled node type {type(node)}")

    def gen_call(self, node: Call) -> str:
        # Handle built-ins mapping here
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
        return f"{node.name}({', '.join(self.gen_expr(a) for a in node.args)})"

    def gen_expr(self, expr: ASTNode) -> str:
        if isinstance(expr, Number):
            return expr.value
        if isinstance(expr, String):
            return expr.value
        if isinstance(expr, Bool):
            return expr.value
        if isinstance(expr, ImplicitVar):
            return "_last_val"
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, ListLit):
            return f"[{', '.join(self.gen_expr(x) for x in expr.items)}]"
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
    # run in fresh globals
    globals_dict: Dict[str, Any] = {}
    try:
        # compile then exec to get proper line numbers in traceback (we still map)
        compiled = compile(py_code, filename="<stubx_generated>", mode='exec')
        exec(compiled, globals_dict)
    except Exception as e:
        # extract tb last frame where filename == <stubx_generated>
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
                # try show the source line (if file available)
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
# CLI
# ---------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Stubx transpiler (Phase 1).")
    p.add_argument('--file', '-f', required=True, help='Stubx source file (.stubx)')
    p.add_argument('--no-run', action='store_true', help='Only compile to .py, do not run')
    args = p.parse_args(argv)

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
    print(f"Compiled {src_path} -> {out_path}")

    if args.no_run:
        print("Compilation finished (no run).")
        return

    print("=== Generated Python (first 200 lines) ===")
    for i, line in enumerate(py.splitlines()[:200], start=1):
        mapped = line_map.get(i)
        m = f"  # <- Stubx ln {mapped[0]}" if mapped else ""
        print(f"{i:04d}: {line}{m}")
    print("=== Running... ===")
    run_generated(py, line_map, src_path)


if __name__ == '__main__':
    main()
