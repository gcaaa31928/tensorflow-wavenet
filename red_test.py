from time import sleep

from wavenet import AudioReader


def main():
    reader = AudioReader(
        './input',
        './output',
        None,
        16000,
        sample_size=10000,
        silence_threshold=0.3)

    reader.start_threads(None)



if __name__ == "__main__":
    # execute only if run as a script
    main()
    sleep(10000)