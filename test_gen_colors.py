"""Test generation info colors"""
from pathlib import Path
from plagih.util import setup_logging, printpl, printez

# Setup logging
setup_logging(console_level=20, verbose=False)

print("=== Testing Generation Info Colors ===\n")

# Test generation info (should be magenta with [Gen] prefix)
printpl('gg', 'Preparing to create first Generation. Gen 0.')
printpl('ggg', '->Evolving 10x \'init_rand1a\'...')
printpl('gggg', '|->15: Add(Symbol(a), Number(2))')

print("\n=== Testing File Write Color ===\n")

# Test file write (should have "Writing File: " prefix, no special color)
printez('f', 'Performance plot saved: C:\\Users\\Simon\\file.png')

print("\n=== Testing Other Colors (for reference) ===\n")

# Test other colors
printpl('i', 'Info message (cyan)')
printpl('a', 'Action completed (green)')
printpl('w', 'Warning message (yellow)')

print("\n=== Test Complete ===")
