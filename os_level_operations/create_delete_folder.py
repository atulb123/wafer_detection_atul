import shutil, os


class CreateDeleteFolder:
    def create_folders(self, folder_names):
        try:
            for folder_name in folder_names:
                if os.path.isdir(os.getcwd() + "/train_test_files/" + folder_name):
                    shutil.rmtree((os.getcwd() + "/train_test_files/" + folder_name))
                    os.mkdir(os.getcwd() + "/train_test_files/" + folder_name)
                else:
                    os.mkdir(os.getcwd() + "/train_test_files/" + folder_name)
        except:
            raise BaseException("Failed to create Good and Bad Folder")

    def clean_and_create_folder(self, path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            os.mkdir(path)

    def perform_folder_clean_up(self):
        base_path = [os.getcwd() + "/train_test_files", os.getcwd() + "/trained_models"]
        for path in base_path:
            for folder_name in list(os.walk(path))[0][1]:
                if folder_name not in ['default_test_files', 'default_training_files']:
                    shutil.rmtree(path + "/" + folder_name)
                    os.mkdir(path + "/" + folder_name)

    def copy_files_to_folder(self, src_path, dest_path):
        for file_name in list(os.walk(src_path))[0][-1]:
            shutil.copy(src_path + "/" + file_name, dest_path + "/" + file_name)
