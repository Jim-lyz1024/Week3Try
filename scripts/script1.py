import os

# program_directory = "/data/dzha866/Project/VIGIL/"
# program_directory = "/home/yil708/data/Week3Try/"
program_directory = "/data/yil708/Code-VIGIL/Week3Try/"
os.chdir(program_directory)

gpu = 2

# dataset = "Digits"
# source_domains = "mnist_m svhn syn"
# target_domains = "mnist"

# dataset = "PACS"
# source_domains = "cartoon photo sketch"
# target_domains = "art_painting"

# dataset = "OfficeHome"
# source_domains = "clipart product real_world"
# target_domains = "art"

# dataset = "VLCS"
# source_domains = "caltech pascal sun"
# target_domains = "labelme"

dataset = "TerraInc"
source_domains = "location_43 location_46 location_100"
target_domains = "location_38"

# dataset = "NICO"
# source_domains = "dim grass outdoor rock water"
# target_domains = "autumn"

# dataset = "DomainNet"
# source_domains = "infograph painting quickdraw real sketch"
# target_domains = "clipart"

backbone = "ViTB32"
model = "CLIPAdapters"
output_dir = "output/" + model + "-" + backbone + "-" + dataset + "-" + target_domains

model_config_file = "config/clipadapters.yaml"

# seeds = [134, 232, 607, 779, 995]
seeds = [134]

for seed in seeds:
    command = (
        "python train.py --gpu {} --seed {} --output-dir {} --dataset {} --source-domains {} --target-domains {}\
            --model {} --model-config-file {}".format(
            gpu,
            seed,
            output_dir,
            dataset,
            source_domains,
            target_domains,
            model,
            model_config_file,
        )
    )
    # os.system("clear")
    print(command)
    os.system(command)
