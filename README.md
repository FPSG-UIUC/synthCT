# SynthCT

Repository for SynthCT framework (NDSS'22). SynthCT uses program synthesis to
automatically synthesize safe translations for unsafe instructions in a target
microarchitecture.

For technical details, please read our paper
[here](https://sushant94.me/publications/22ndss-synthct.pdf)

Interested in synthesized translations? Look
[here](https://github.com/FPSG-UIUC/synthCT-artifacts) for an open-source release of
translations synthesized in our experiments!

This is a research prototype, so please be gentle :smile:
If there are glaring bugs that prevent you from running the prototype, please open an
issue and we will try to fix it ASAP.

Interested in contributing? Please get in touch!

## Quick Start Guide

In this section, we will quickly setup SynthCT and synthesize our first instruction!

We recommend using [docker](https://docs.docker.com/get-docker/) for quick setup.

### Docker Setup Guide

1. Install docker following instructions from the docker guide.
2. Clone this repository
3. Initialize and clone the submodules
4. Run docker build script: `./build_docker.sh`
5. Create a directory to save synthesis results: `mkdir synth_results`
6. Start the docker image:
`docker run -it /<absolute_path>/synth_results/:/synthesis/synthCT/synth-results synthesis:latest`

To synthesis an example instruction, e.g., add-with-carry (ADCQ):

`python3 -m synthesis.synthesis \
  --isa ./third-party/x86-64-semantics/semantics/registerInstructions/*.k \
  --only "ADCQ-R64-R64" \
  --pseudo-inst ./synthesis/pseudo.yaml \
  --selector "knn" \
  --parallel-tasks 2 \
  --timeout 1200`

Synthesis is designed to run forever, trying to generate multiple different solutions.
Once a translation has been found, use CTRL+C to terminate synthesis and check
`synth_results` directory for translation.

That's it! You just synthesized your first x86 instruction translation. Follow along the
rest of the guide to customize your synthesis runs.

## Setup Guide

Requirements:

1. Install racket and rosette using the steps from this [guide](https://emina.github.io/rosette/)
2. >= python3.6

Steps:

1. Clone the repository
2. Run setup.sh: `./setup.sh`
   This step will setup the dependencies and create a new python3 virualenv named: `.synth-venv`
3. Activate the virtual-env: `source ./.synth-venv/bin/activate`

Try executing the top-level synthesis command to check if everything works fine:
`python3 -m synthesis.synthesis --help`

If you see the help menu, you're all set!

## Controlling Synthesis

This section goes through the various options to control and fine-tune the synthesis
process. Please check quickstart guide for a default example in running synthesis.

To see help, run the top-level synthesis script: `python3 -m synthesis.synthesis -h`

Here are some useful options:

* `--parallel-tasks N` : Number of parallel synthesis tasks to run. This is ideally equal to
  number of core. Larger number can run more instruction synthesis tasks in parallel.

* `--isa <ISA>` : Can be used to control the subset of ISA available for synthesis. Full
  list of instructions semantics are under the
  `./third-party/x86-64-semantics/semantics/registerInstructions/` directory. This flag
  takes multiples (space-separated) files as arguments.

* `--only <I1> <I2> ...` : Can be used to control which instructions to synthesize. Space
  separated instruction names to add to the synthesis queue.

* `--pseudo-inst <yaml>` : YAML file containing a list of pseudo instructions to use for
  synthesis. See the default file: `synthesis/pseudo.yaml` for an example.

* `--selector <name>` : Component selection strategy. Paper uses 'knn' (the default). Feel
  free to explore other component selection strategies.

* `--try-instruction-factorizarion`: Flag that enables instruction factorization when
  initial synthesis attempt(s) fail.

* `--factorizer <NAME>` : Instruction factorization strategy. Paper uses 'bottom\_up'
  (default). Only used when factorization is enabled. See paper for details on instruction
  factorization.

* `--iterative-rewrites` : Enables iterative node-splitting. Only used for synthesis of
  division family of instructions.


## Distributed Queues and Persistent Storage

By default, synthesis using python queues to distribute work and get results back, and
dumps synthesis results to yaml files.
This is good for quick, simple, local synthesis runs but falls short for long synthesis runs.
Fortunately, synthCT supports long-running distributed synthesis tasks.

To set this up, first setup [beanstalkd](https://beanstalkd.github.io/) for communication and
[mongodb](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/) for saving
results. To run synthesis in this mode, start synthesis with the parameter: `--conn <CONN>`.
The `<CONN>` file is yaml configuration file that specifies `host` and `port` for the
queue and the database. Check `docker/pipe_config.yaml` for an example.

Note: Make sure that the services are running and accessible from machines you are running
synthesis tasks from before starting the run!

## Cite

If you find our work helpful, the framework or the artifacts, please cite our NDSS'22
paper:

```
@inproceedings{synthct,
  author    = {Sushant Dinesh and
               Grant Garrett-Grossman and
               Christopher W. Fletcher
  },
  title     = {SynthCT: Towards Portable Constant-Time Code},
  booktitle = {{NDSS}},
  publisher = {The Internet Society},
  year      = {2022}
}
```

## Licence

MIT License

Copyright (c) 2022 Sushant Dinesh <sushant.dinesh94@gmail.com> and FPSG-UIUC group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Dev Notes

TODO
