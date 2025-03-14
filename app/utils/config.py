from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # db
    mongo_uri: str

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()
