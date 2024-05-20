import os

# program_directory = "/data/dzha866/Project/VIGIL/"
program_directory = "/home/yil708/data/Week3Try/"
os.chdir(program_directory)

gpu = 0

# dataset = "Digits"
# source_domains = "mnist mnist_m svhn"
# target_domains = "syn"

# dataset = "PACS"
# source_domains = "art_painting cartoon photo"
# target_domains = "sketch"

# dataset = "OfficeHome"
# source_domains = "art clipart product"
# target_domains = "real_world"

# dataset = "VLCS"
# source_domains = "caltech labelme pascal"
# target_domains = "sun"

# dataset = "TerraInc"
# source_domains = "location_38 location_43 location_46"
# target_domains = "location_100"

dataset = "NICO"
source_domains = "autumn dim grass outdoor water"
target_domains = "rock"

# dataset = "DomainNet"
# source_domains = "clipart infograph painting quickdraw sketch"
# target_domains = "real"

backbone = "ViTB32"
model = "CLIPAdapters"
output_dir = "output/" + model + "-" + backbone + "-" + dataset + "-" + target_domains

model_config_file = "config/clipadapters.yaml"

seeds = [134, 232, 607, 779, 995]

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
