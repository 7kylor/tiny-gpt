# Tiny LM

A simple language model in C++23. It generates readable text in seconds.

Inspired by nanoGPT and Andrej Karpathy's approach - start simple and make it work. I wanted to understand how LLMs work under the hood, so I built this tiny model.

This is for learning and research purposes only.

## Quick Start

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

./tiny_lm --data ../data/tinyshakespeare.txt
```

## Sample Output

After training for a few seconds:

```
ROMEO:
The of with danger and not a man and
And beat her words,
Why, how he does your good

GLOUCESTER:
Pray God save yours, and you, good
My lord,

ISABELLA:
This widow, field God, blood which I for his state,
```

It generates real words, character names, and dialogue structure. Pretty readable for something so simple.

## How It Works

It's a word-level bigram model - basically learns P(next_word | current_word).

Why word-level instead of character-level? Words are already meaningful units, no need to learn spelling, output is immediately readable, and it trains in seconds.

## Parameters

- `--data` - Training text file (default: data/tinyshakespeare.txt)
- `--vocab` - Vocabulary size (default: 3000)
- `--steps` - Training steps (default: 5000)
- `--lr` - Learning rate (default: 1.0)

## Examples

```bash
# Quick training
./tiny_lm --steps 2000

# Longer training for better quality
./tiny_lm --steps 20000 --lr 0.3

# Larger vocabulary
./tiny_lm --vocab 5000 --steps 15000
```

## Data

Uses tinyshakespeare.txt from Karpathy's char-rnn repo.

## Requirements

- C++23 compiler
- CMake 3.20+

## Why This Works

Character-level transformers need correct backpropagation through attention, thousands of training steps, and ideally a GPU.

Word-level bigrams just need simple gradient descent, hundreds of steps, and CPU is fine. The output quality is surprisingly good for the effort.

## Credits

- Andrej Karpathy for nanoGPT and makemore
- Attention Is All You Need paper

## License

MIT
