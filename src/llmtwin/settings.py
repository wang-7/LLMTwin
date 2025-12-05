from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # MongoDB
    DATABASE_NAME: str = 'mongodbVSCodePlaygroundDB'

settings = Settings()