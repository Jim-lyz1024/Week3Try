import os

program_directory = "/data/dzha866/Project/VIGIL/"
os.chdir(program_directory)

gpu = 1

# dataset = "Digits"
# source_domains = "mnist svhn syn"
# target_domains = "mnist_m"

# dataset = "PACS"
# source_domains = "art_painting photo sketch"
# target_domains = "cartoon"

# dataset = "OfficeHome"
# source_domains = "art product real_world"
# target_domains = "clipart"

# dataset = "VLCS"
# source_domains = "caltech pascal sun"
# target_domains = "labelme"

# dataset = "TerraInc"
# source_domains = "location_38 location_46 location_100"
# target_domains = "location_43"

# dataset = "NICO"
# source_domains = "autumn grass outdoor rock water"
# target_domains = "dim"

dataset = "DomainNet"
source_domains = "clipart painting quickdraw real sketch"
target_domains = "infograph"

backbone = "ViTB32"
model = "CoOp"
output_dir = "output/" + model + "-" + backbone + "-" + dataset + "-" + target_domains

model_config_file = "config/coop.yaml"

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
    os.system("clear")
    print(command)
    os.system(command)
