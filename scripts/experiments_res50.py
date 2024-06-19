import os

# program_directory = "/home/yil708/data/Week3Try/"
program_directory = "/data/yil708/Code-VIGIL/Week3Try/"
os.chdir(program_directory)

gpu = 3

# Define the datasets and domains to be used
datasets = ['PACS', 'OfficeHome', 'VLCS', 'TerraInc', 'NICO']
domains = {
    # 'Digits': ['mnist', 'mnist_m', 'svhn', 'syn'],
    'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'OfficeHome': ['art', 'clipart', 'product', 'real_world'],
    'VLCS': ['caltech', 'labelme', 'pascal', 'sun'],
    'TerraInc': ['location_38', 'location_43', 'location_46', 'location_100'],
    'NICO': ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
}

# Define the models to be evaluated
models = ['CLIPAdapters']

# Define source domains for each dataset
source_domains = {
    # 'Digits': ['mnist', 'mnist_m', 'svhn', 'syn'],
    'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'OfficeHome': ['art', 'clipart', 'product', 'real_world'],
    'VLCS': ['caltech', 'labelme', 'pascal', 'sun'],
    'TerraInc': ['location_38', 'location_43', 'location_46', 'location_100'],
    'NICO': ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
}

def main(): 
    # Loop over datasets, domains, and models
    for dataset_name in datasets:
        for domain_name in domains[dataset_name]:
            for model_name in models:
                backbone = "RN50"

                output_dir = "output/backbonesnew/" + model_name + "-" + backbone + "-" + dataset_name + "-" + domain_name

                model_config_file = "config/clipadapter.yaml" if model_name == 'CLIPAdapter' else "config/clipadapters.yaml"
                
                source_domains_list = [d for d in source_domains[dataset_name] if d != domain_name]
                source_domains_str = ' '.join(source_domains_list)
                
                command = (
                    "python train.py --gpu {} --seed {} --output-dir {} --dataset {} --source-domains {} --target-domains {}\
                        --model {} --model-config-file {}".format(
                        gpu,
                        134,
                        output_dir,
                        dataset_name,
                        source_domains_str,
                        domain_name,
                        model_name,
                        model_config_file,
                    )
                )
                # os.system("clear")
                print(command)
                os.system(command)

if __name__ == '__main__':
    main()
