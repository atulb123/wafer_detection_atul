import os
class ReadFolder:
    def return_list_of_file(self,path):
        try:
            return list(os.walk(path))[0][-1]
        except:
            raise BaseException("Failed to read file/Path Doesn't exists")
