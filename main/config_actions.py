import configparser

def config_write(db_path, config_path="database.ini"):
    '''Записывает в конфиг путь до стартовой БД.'''
    config = configparser.ConfigParser()
    config.read(config_path)
    #config.add_section("SETTINGS")
    config.set("SETTINGS", "start_db", db_path)
    with open(config_path, "w") as config_file:
        config.write(config_file)

def config_read(filename="database.ini"):
    '''Читает путь до стратовой БД.'''
    config = configparser.ConfigParser()
    config.read(filename)
    return config.get("SETTINGS", "start_db")

#config_write('database.db')
