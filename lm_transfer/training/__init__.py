import random
import string


def generate_experiment_id(length=6):
    characters = string.ascii_letters + string.digits
    experiment_id = ''.join(random.choice(characters) for _ in range(length))
    return experiment_id
