"""Purge all files from the trials directory. (BE CAREFUL THIS WILL OVERWRITE IMPORTANT STUFF IF NOT CAREFUL)"""

import os
import shutil


def purge_trials():
    """Delete all files in the trials directory"""

    trials_dir = "trials"

    if not os.path.exists(trials_dir):

        print(f"'{trials_dir}' directory does not exist. Nothing to purge.")

        return

    files = [f for f in os.listdir(trials_dir) if os.path.isfile(os.path.join(trials_dir, f))]

    file_count = len(files)

    if file_count == 0:

        print(f"'{trials_dir}' directory is already empty.")

        return

    print(f"About to delete {file_count} files from '{trials_dir}' directory.")

    response = input("Continue? (yes/no): ").strip().lower()

    if response not in ["yes", "y"]:

        print("Purge cancelled.")

        return

    deleted = 0

    for filename in files:

        filepath = os.path.join(trials_dir, filename)

        try:

            os.remove(filepath)

            deleted += 1

        except Exception as e:

            print(f"Error deleting {filename}: {e}")

    print(f"Deleted {deleted} files from '{trials_dir}' directory.")


if __name__ == "__main__":

    purge_trials()
