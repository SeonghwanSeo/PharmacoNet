from pmnet_appl.sbddreward import SBDDReward_Proxy

docking = "UniDock_Vina"
train_dataset = "ZINC"
proxy = SBDDReward_Proxy.load(docking, train_dataset, None, "cuda")

save_database_path = "./tmp_db.pt"
protein_info_dict = {
    "key1": ("./tmp1.pdb", "./ref_ligand1.sdf"),  # reference ligand path
    "key2": ("./tmp2.pdb", (1.0, 2.0, 3.0)),  # pocket center
}

cache_dict = proxy.get_cache_database(protein_info_dict, save_database_path, verbose=False)
proxy.update_cache(cache_dict)
proxy.scoring(list(cache_dict.keys())[0], "c1ccccc1")
proxy.scoring_list(list(cache_dict.keys())[0], ["c1ccccc1", "C1CCCCC1"])
