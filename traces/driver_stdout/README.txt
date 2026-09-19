Raw stdout of the drivers, as it was produced. One line per request.

frozen_unfixed.txt            frozen_ab.py, main 71fc70d3, align, prefix on
frozen_fixed.txt              frozen_ab.py, + the #47123 runner hunks
frozen_unfixed_replicate.txt  frozen_ab.py, unfixed again on a second server.
                              The label in this file says "unfixed-mamba-none"
                              because the server was started with
                              --mamba-cache-mode none; the engine overrode that
                              back to align (README.md section 6), so it is an
                              independent replicate of the unfixed arm, and that
                              is how it is labelled in results/.
mamba_none.txt                the genuine none-mode cell (prefix caching off)
preempt_*.txt                 preempt_probe.py; each run prints the
                              vllm:num_preemptions_total delta on its DONE line
