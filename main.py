import argparse

from extract_ecg_clean import process_folder as process_screen_folder
from extract_ecg_scan_clean import process_folder as process_scan_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="screen", choices=["screen", "scan"])
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.mode == "scan":
        process_scan_folder(args.input_dir, args.output_dir, debug=args.debug)
    else:
        process_screen_folder(args.input_dir, args.output_dir, debug=args.debug)


if __name__ == "__main__":
    main()
