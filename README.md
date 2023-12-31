# NVFLARE CIFAR10 Provisioning END-to-END with Inference

## OS Ubuntu 22.04

### follow the installer in environment_setup.txt, for the cuda installation there are several ways, remember to install the 11.8 version.

- project folder located in workspaces/secure_workspaces
- add to /admin@nvidia.com the "local" folder, required by nvflare.fuel.hci.tools.admin


## Relevant Features Change

- every client must have resources.json file in the local folder, this file can be inherited from resources.json.default ( remember to change the gpu mem and num of gpus from 0 ( for example to 1)):   
<code>
      "args": {
        "num_of_gpus": 1,
        "mem_per_gpu_in_GiB": 1
      }
    </code>  

- in startup/sub_start.sh file change the PYTHONPATH ( line 5) to :
<code>export PYTHONPATH="${PYTHONPATH}:${PWD}/../local/custom"</code>
required to find the learners etc..
- Each client, server must have a custom/pt in their local folder with CIFAR-10-images, learners, network, utils.
- move jobs to transfer folder in admin@nvidia.com/

## Download and set-up the CIFAR10 data

To download the CIFAR10 data run the script:

<code>./download_data.sh</code>

After that put the generated folder in the following directories:
- /localhost_text1/prod_00/localhost/local/custom/pt
- /localhost_text1/prod_00/site-1/local/custom/pt
- /localhost_text1/prod_00/site-2/local/custom/pt

Remember to keep a copy for the CIFAR10_inference.ipynb in the main directory

## Custom/pt

What you will find in the custom/pt folder is a modified version of the original nvflare_cifar10 by nvidia.
Remember to put the CIFAR-10-images folder in the custom/pt of each client, server.
