import logging
from core import ShmManager, process_and_analyze

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

def main():
    print("Starting server...")
    with ShmManager() as shm:
        for frame in shm.read():
            result = process_and_analyze(frame.bgr, show_box=True)

            shm.write(result, frame.frame_id)


if __name__ == "__main__":
    main()
