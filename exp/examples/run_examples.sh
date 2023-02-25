# Script for executing all examples. Examples match the pattern "example_*.py".

set -e

for example in example_*.py; do
    printf "\n##############################################################"
    printf "\nRunning python $example"
    printf "\n##############################################################"
    printf "\n\n"
    python $example
done
