import os

# program_directory = "/data/dzha866/Project/VIGIL/"
program_directory = "/home/yil708/data/Week3Try/"
# program_directory = "/data/yil708/Code-VIGIL/Week3Try/"
os.chdir(program_directory)

gpu = 3

# dataset = "Digits"
# source_domains = "mnist mnist_m syn"
# target_domains = "svhn"

# dataset = "PACS"
# source_domains = "art_painting cartoon sketch"
# target_domains = "photo"

dataset = "OfficeHome"
source_domains = "art clipart real_world"
target_domains = "product"

# dataset = "VLCS"
# source_domains = "caltech labelme sun"
# target_domains = "pascal"

# dataset = "TerraInc"
# source_domains = "location_38 location_43 location_100"
# target_domains = "location_46"

# dataset = "NICO"
# source_domains = "autumn dim outdoor rock water"
# target_domains = "grass"

# dataset = "DomainNet"
# source_domains = "clipart infograph quickdraw real sketch"
# target_domains = "painting"

backbone = "ViTB32"
# backbone = "RN50"
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
