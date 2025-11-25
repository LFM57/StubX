# Stubx: The Poetic and Powerful Scripting Language

Stubx is an **intuitive, elegant, and modern programming language** designed for unparalleled clarity and power. It aims to make coding feel more like writing prose, focusing on the logic and ideas rather than complex syntax.

## Key Features

*   **Readability First:** Minimal syntax reduces clutter, making your code easy to understand for developers and non-developers alike.
*   **Pipes & Juxtaposition:** Effortlessly chain operations and call functions. Write your logic from left to right, enhancing flow and comprehension.
*   **Implicit Value (`~`):** Access the result of the last operation instantly, allowing for a seamless thought process in your code.
*   **Transpiles to Python:** Stubx leverages the robustness and extensive ecosystem of Python, generating clean and readable Python code.
*   **Intuitive Control Structures:** Clear `if/else`, `while`, and `for` loops with block definitions using `:` and `end`.
*   **Flexible Typing:** Dynamically typed with optional suffixes (`n` for number, `s` for string, `?` for boolean) for precise control when needed.

## Quick Start

To compile and run a Stubx script (`.stubx` file):

```bash
python compilator.py --file your_script.stubx
```

To compile only (this will generate a `.py` file):

```bash
python compilator.py --file your_script.stubx --no-run
```

## Example

```stubx
-- A simple Stubx program
ask username
say "Hello, " + username + "!"

if username == "Louis":
    say "Welcome back, creator!"
else:
    say "Nice to meet you."
end

-- Using pipes and implicit value
"stubx is fun" |> upper |> say
-- Output: STUBX IS FUN

10 + 20
say "The result is: " + ~
-- Output: The result is: 30
```

## About the Project

Stubx was created by **Louis Simonet** with a goal to create a more accessible and expressive program language. The project aims to simplify how we write code, bridging the gap between natural language and powerful scripting.

## Links

*   **Main page:** [Welcome page](https://louis.simonet.name/other/StubX/)
*   **Documentation:** [Detailed Stubx Documentation](https://louis.simonet.name/other/StubX/docs.html)
*   **Download Compiler from an other source:** [compilator.py](https://louis.simonet.name/other/StubX/compilator.py)

## Disclaimer

Beware, this project is still in development and the main features you might expect probably aren't implemented yet.
I am activly working on it, but I can't garantee further development
