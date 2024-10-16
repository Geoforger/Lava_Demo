import numpy as np
import threading
import time
from core.sensor.tactile_sensor_neuro import NeuroTac


def make_sensor():
    return NeuroTac(save_events_video=False, save_acc_video=False, display=False)


def collect_data(output_path, labels=[0, 1, 2]):
    n_trials = 10

    with make_sensor() as sensor:
        label_idx = input(f"Select from label from: {labels}...")

        for trial_idx in range(n_trials):
            events_on_file = f"{output_path}/{label_idx}-{trial_idx}_on.pickle"
            events_off_file = f"{output_path}/{label_idx}-{trial_idx}_off.pickle"
            sensor.set_filenames(
                events_on_file=events_on_file, events_off_file=events_off_file
            )

            sensor.reset_variables()

            #  Start sensor recording when sliding
            sensor.start_logging()
            t = threading.Thread(target=sensor.get_events, args=())
            print(f"Starting collection iteration {trial_idx}...")
            input("Press any button to start: ")
            t.start()

            # Slide - Blocking movement
            time.sleep(2)

            # Stop sensor recording after sliding
            sensor.stop_logging()
            t.join()
            print("Stopped collection...")

            # Collate proper timestamp values in ms.
            sensor.value_cleanup()

            # Save data
            sensor.save_events_on(events_on_file)
            sensor.save_events_off(events_off_file)
            print("Saved data!")


def main():
    output_path = "/media/farscope2/My Passport/George/George Datasets/lava_demo/"
    collect_data(output_path)


if __name__ == "__main__":
    main()
