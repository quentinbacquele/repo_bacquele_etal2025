import subprocess
import os
import sys # To potentially suggest where sumtrees might be

# --- Configuration ---

# Input file containing MULTIPLE TREES (e.g., 1000)
input_tree_file = "./data/AllBirdsEricson1.tre"

# Output file for the consensus tree (Newick format with branch lengths)
# Using the same output directory and base naming as the R script for consistency
output_consensus_file = "./output/consensus_pruned_tree_plots/consensus_sumtrees.tre"

# --- Advanced Configuration (Usually OK as default) ---

# Path to sumtrees.py executable.

sumtrees_executable = "./.venv/bin/sumtrees.py"

# Consensus level (e.g., 0.5 for 50% majority rule)
majority_rule_threshold = 0.5

# Branch length calculation method ('mean' or 'median')
branch_length_method = "mean"

# --- End Configuration ---

# Ensure output directory exists
output_dir = os.path.dirname(output_consensus_file)
os.makedirs(output_dir, exist_ok=True)
print(f"Ensured output directory exists: {output_dir}")

# Check if input file exists
if not os.path.exists(input_tree_file):
    print(f"Error: Input tree file not found at '{input_tree_file}'")
    sys.exit(1) # Exit with error code

# --- Construct the SumTrees Command ---
# Build the command as a list of arguments using UPDATED flags
sumtrees_command = [
    sumtrees_executable,
    # Input processing options
    "--rooted",                            # Assume input trees are rooted
    "--ultrametric",                       # Expect ultrametric
    "--ultrametricity-precision=10000000", # Effectively bypass strict check

    # Consensus definition options
    # '-f 0.5' is usually the default for majority rule, only needed if changing
    # Add '-f 0.5' here to be explicit or use a different value, e.g., '-f 0.95'
    # '-f', '0.5', # Example if explicitly setting 50%

    # Branch length options - UPDATED FLAG
    '-e', 'mean-length',                   # Use mean edge lengths

    # Output options - UPDATED FLAG
    "--output=" + output_consensus_file,
    '-F', 'newick',                        # Output format

    # Force overwrite if output exists? Add --force if needed.
    # '--force', # Uncomment if needed

    # Input file (last argument)
    input_tree_file
]

# --- Run SumTrees ---
print("\nCalculating consensus tree with branch lengths using SumTrees...")
print("-" * 60)
print("Command to be executed:")
# Print command in a way that's easy to copy/paste into a terminal for debugging
print(" ".join(f'"{arg}"' if " " in arg else arg for arg in sumtrees_command))
print("-" * 60)

try:
    # Run the command, capture output, check for errors, decode output as text
    process = subprocess.run(
        sumtrees_command,
        check=True,           # Raise an exception if SumTrees returns non-zero exit code
        capture_output=True,  # Capture stdout and stderr
        text=True,            # Decode output as text (UTF-8 default)
        encoding='utf-8'      # Be explicit about encoding
        )

    # Print outputs (SumTrees often prints info/progress to stderr)
    print("\nSumTrees Standard Output:")
    print(process.stdout if process.stdout else "(No stdout)")
    print("\nSumTrees Standard Error:")
    print(process.stderr if process.stderr else "(No stderr - check progress above or if run finished immediately)")
    print("-" * 60)
    print(f"SumTrees completed successfully.")
    print(f"Consensus tree saved to: {output_consensus_file}")

except FileNotFoundError:
    # Error if sumtrees_executable wasn't found
    python_executable_dir = os.path.dirname(sys.executable)
    print(f"\nError: Could not find the command '{sumtrees_executable}'.")
    print("Please ensure DendroPy is installed correctly (`pip install dendropy`)")
    print("and that 'sumtrees.py' is in the system's PATH.")
    print(f"(Python executable is in: {python_executable_dir})")
    print(f"If needed, manually set the full path to 'sumtrees.py' in the 'sumtrees_executable' variable within this script.")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    # Error if SumTrees ran but returned an error code
    print("\nError: SumTrees failed to run correctly.")
    print("Return Code:", e.returncode)
    print("\nSumTrees Standard Output:")
    print(e.stdout if e.stdout else "(No stdout)")
    print("\nSumTrees Standard Error (contains error details):")
    print(e.stderr if e.stderr else "(No stderr)")
    sys.exit(1)
except Exception as e:
    # Catch other unexpected errors
    print(f"\nAn unexpected error occurred: {e}")
    sys.exit(1)

print("\nPython script finished.")