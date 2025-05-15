# Cue2: a deep learning framework for SV calling and genotyping

##### Table of Contents   
[Installation](#install)  
[User Guide](#guide)  
[Recommended workflow](#workflow)    


### Installation

1\. Clone the repository: ```$> git clone git@github.com:PopicLab/cue2.git```

2\. Navigate into the ```cue2``` folder: ```$> cd cue2```

3\. Setup a Python virtual environment (recommended)
- Create the virtual environment (in the env directory): ```$> python3.9 -m venv env```
- Activate the environment: ```$> source env/bin/activate```

4\. Install the framework:
```$> pip install .```

5\. Set the ```PYTHONPATH```: ```export PYTHONPATH=${PYTHONPATH}:/path/to/cue2```

6\. Download the latest pre-trained Cue models from this [Google Cloud Storage bucket](https://console.cloud.google.com/storage/browser/cue-models)


<a name="guide"></a>
### User guide

#### Execution

* To call structural variants: ```$> cue call --config </path/to/config>```  
* To train a new model: ```$> cue train --config </path/to/config>```
* To generate a training dataset: ```$> cue generate --config </path/to/config>```

Each ```cue``` command accepts a YAML file with configuration parameters. Template config files are provided in the 
```config/``` directory. 

The key parameters for each ```cue``` command are listed below.

```call```:
* ```bam``` [*required*] path to the alignments file (BAM/CRAM format)
* ```fai``` [*required*] path to the reference FASTA FAI file
* ```chr_names``` [*optional*] list of chromosomes to process: null (all) or a specific list e.g. ["chr1", "chr21"] (default: null)
* ```model_path``` [*required*] path to the pretrained Cue model (recommended: the latest available model)
* ```gpu_ids``` [*optional*] list of GPU ids to use for calling (default: CPU(s) will be used if empty)
* ```n_jobs_per_gpu``` [*optional*] number of parallel jobs to launch on the same GPU (default: 1)
* ```n_cpus```  [*optional*] number of CPUs to use for calling if no GPUs are listed (default: 1)

```train```:
* ```dataset_dirs``` [*required*] list of annotated imagesets to use for training
* ```gpu_ids```  [*optional*] GPU id to use for training -- a CPU will be used if empty
* ```report_interval``` [*optional*] frequency (in number of batches) for reporting training stats and image predictions (default: 50)

```generate```:
* ```bam``` [*required*] path to the alignments file (BAM/CRAM format)
* ```vcf``` [*required*] path to the ground truth SV BED or VCF file
* ```fai``` [*required*] path to the reference FASTA FAI file
* ```n_cpus```  [*optional*] number of CPUs to use for image generation (parallelized by chromosome) (default: 1)
* ```chr_names``` [*optional*] list of chromosomes to process: null (all) or a specific list e.g. ["chr1", "chr21"] (default: null)


<a name="workflow"></a>
#### Recommended workflow 

1. Create a new directory for the experiment.
2. Place the YAML config file in this directory (see the provided templates).
3. Populate the YAML config file with the parameters specific to this experiment.
4. Execute the appropriate ```cue``` command providing the path to the newly configured YAML file.
```cue``` will automatically create auxiliary directories with results in the folder where the config YAML file is located.

