This probes our optimizations for the linear layer and compares them to other
approaches. The results are described in the appendix, where details on the
optimization are laid out.

**Note:** You have to install `vivit-for-pytorch`:
```bash
pip install vivit-for-pytorch==1.0.0
```
Then, run the script
```bash
python run_comparison.py
```
which will print the run time results in the terminal.

*Important:* Don't forget to re-install the research version of `vivit` afterwards:
```bash
cd ../.. && pip install -e .
```
