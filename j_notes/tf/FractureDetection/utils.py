from operator import methodcaller


class Config:
    def __init__(self):
        self.options = self.read_config()
        self.data_dir = self.options["data_dir"]

    def read_config(self, file_name="config"):
        with open(file_name, "r") as fd:
            lines = fd.readlines()
        lines = map(methodcaller("strip"), lines)
        lines = map(methodcaller("split", ";"), filter(lambda l: not l.startswith("#"), lines))
        options = {l[0]: l[1] for l in lines}
        return options


if __name__ == "__main__":
    config = Config()
    print(config.options)