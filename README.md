# Stubx: The Poetic and Powerful Scripting Language

Stubx is an **intuitive, elegant, and modern programming language** designed for unparalleled clarity and power. It aims to make coding feel more like writing prose, focusing on the logic and ideas rather than complex syntax.

## Versions & Features

### First Version (V1) - The Foundation
The core capabilities of Stubx:
*   **Readability First:** Minimal syntax reduces clutter.
*   **Pipes & Juxtaposition:** Chain operations (`x |> f`) and call functions naturally.
*   **Implicit Value (`~`):** Access the result of the last operation instantly.
*   **Control Structures:** Clear `if/else`, `while`, and `for` loops.
*   **Flexible Typing:** Dynamically typed with optional suffixes.

### Stable Release (V2) - Extended Capabilities
Adds powerful system integration and functional patterns:
*   **File System:** Easily `read` from and `append_to` files.
*   **System Interaction:** Execute shell commands directly with `exec`.
*   **Functional Power:** Use `map` and `filter` for concise list manipulations.
*   **Error Handling:** Handle exceptions gracefully with `attempt` and `recover` blocks.

### Current Release (V3) - Developer Tools (New!)
Focuses on tooling, interactivity, and debugging:
*   **Interactive REPL:** Experiment with Stubx code in real-time.
*   **Formatter:** Automatically format your code for consistent style.
*   **Debugger Mapping:** Runtime errors in Python are mapped back to their original line in the Stubx source file for easier debugging.

## Quick Start

To run a Stubx script using the latest version (V3):

```bash
python "Current Release (V3)/compiler.py" --file your_script.stubx
```

### Other Commands (V3)

**Start REPL (Interactive Shell):**
```bash
python "Current Release (V3)/compiler.py" --repl
```

**Format Code:**
```bash
python "Current Release (V3)/compiler.py" --format your_script.stubx
```

**Compile Only:**
```bash
python "Current Release (V3)/compiler.py" --file your_script.stubx --no-run
```

## Directory Structure

*   `First Version (V1)/`: The original version of the compiler.
*   `Stable Release (V2)/`: The previous beta, now stable with system features.
*   `Current Release (V3)/`: The latest version including the REPL and Formatter.

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

-- Using V2+ features
attempt:
    read "config.txt"
recover:
    say "Config missing, using defaults."
end
```

## About the Project

Stubx was created by me with a goal to create a more accessible and expressive program language. The project aims to simplify how we write code, bridging the gap between natural language and powerful scripting.

## Links

*   **Main page:** [Welcome page](https://louis.simonet.name/StubX%20Project/)
*   **Documentation:** [Detailed Stubx Documentation](https://louis.simonet.name/StubX%20Project/docs.html)

## Disclaimer

Beware, this project is still in development and the main features you might expect probably aren't implemented yet.
I am actively working on it, but I am a student, so I obviously can't guarantee further development.
