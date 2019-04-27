import argparse
import logging
from typing import Iterable, List, Optional

import pandas as pd


# ----------------------------------------------------------------------------
#       Logging
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#       Commandline parser
# ----------------------------------------------------------------------------

def get_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        help="CSV file with training data."
    )
    parser.add_argument(
        "-c", "--columns", type=str, nargs="+",
        help="Columns in the CSV file to use for training."
    )
    parser.add_argument(
        "--drop-duplicates", type=str, nargs="*",
        help="Drop duplicates in this columns from the CSF file."
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of samples to take from the CSV file."
    )
    parser.add_argument(
        "-e", "--embedding-size", type=int, default=100,
        help="Embedding size."
    )
    parser.add_argument(
        "--context-width", type=int, default=5,
        help="Width of the context window."
    )
    parser.add_argument(
        "--context-samples", type=int, default=1,
        help="Number of samples to take from the context window."
    )
    parser.add_argument(
        "--negative-samples", type=int, default=5,
        help="Negative samples to use in the training."
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Number of steps to move the context window while scanning the input."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size during training."
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of epochs to train."
    )

    return parser


# ----------------------------------------------------------------------------
#       Read data
# ----------------------------------------------------------------------------

def read_data(
        csv_file: str,
        columns: List[str],
        samples: Optional[int] = None,
        drop_duplicates: Optional[List[str]] = None
) -> Iterable[str]:

    logger.info("Reading columns %s from file %s ..." % (columns, csv_file))

    df = pd.read_csv(csv_file)
    df.fillna("", inplace=True)

    if samples is not None:
        df = df.sample(samples)

    if drop_duplicates is not None:
        df .drop_duplicates(drop_duplicates, inplace=True)

    texts: List[str] = []
    for _, row in df[columns].iterrows():
        texts.append(" ".join(field for field in row))

    logger.info("Successfully read the text corpus!")

    return texts


# ----------------------------------------------------------------------------
#       Main
# ----------------------------------------------------------------------------

def main():

    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.resolve()))

    from word2vec.models import SkipGramModel

    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    # Read data.
    texts: List[str] = read_data(
        csv_file=args.csv,
        columns=args.columns,
        samples=args.samples,
        drop_duplicates=args.drop_duplicates
    )

    # Define model.
    model = SkipGramModel(
        embedding_size=args.embedding_size,
        context_window=args.context_width,
        context_samples=args.context_samples,
        negative_samples=args.negative_samples,
        stride=args.stride,
    )

    # Train model.
    model.fit(
        texts=texts,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
