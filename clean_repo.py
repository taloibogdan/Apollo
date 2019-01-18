import os


def remove_placeholders(current_dir):
    for filename in os.listdir(current_dir):
        filepath = os.path.join(current_dir, filename)

        if os.path.isdir(filepath):
            remove_placeholders(filepath)

        if filename == 'placeholder':
            os.unlink(filepath)


if __name__ == '__main__':
    remove_placeholders('.')

