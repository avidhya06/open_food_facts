import pandas as pd
import os
import sys


class DataCleaner:
    def __init__(self, input_file_path, output_json_path, delimiter='\t', missing_threshold=0.7, chunk_size=50000,
                 handle_bad_lines='skip', file_encoding='utf-8'):
        # Initialize the DataCleaner class with file paths, delimiter, thresholds, and other configurations
        self.input_file_path = input_file_path
        self.output_json_path = output_json_path
        self.delimiter = delimiter
        self.missing_threshold = missing_threshold
        self.chunk_size = chunk_size
        self.handle_bad_lines = handle_bad_lines
        self.file_encoding = file_encoding
        self.columns_to_keep = []  # List of columns to retain after filtering
        self.total_rows_processed_pass1 = 0  # Counter for rows processed in Pass 1
        self.total_rows_processed_pass2 = 0  # Counter for rows processed in Pass 2

    def validate_paths(self):
        # Ensure the output directory exists or create it
        output_dir = os.path.dirname(self.output_json_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"Error: Could not create output directory '{output_dir}'. {e}")
                sys.exit(1)

        # Check if the output file path is writable
        try:
            with open(self.output_json_path, 'w') as f_test:
                pass
            print(f"Output path '{self.output_json_path}' is writable.")
        except (IOError, OSError) as e:
            print(f"Error: Cannot write to output path '{self.output_json_path}'. {e}")
            sys.exit(1)

    def identify_columns_to_keep(self):
        # First pass: Identify columns with missing data below the threshold
        print("\n--- Pass 1: Identifying columns to keep ---")
        missing_counts = None  # To store the count of missing values for each column
        column_names = None  # To store column names from the input file

        read_csv_args = {
            'sep': self.delimiter,
            'chunksize': self.chunk_size,
            'low_memory': False,
            'on_bad_lines': self.handle_bad_lines,
            'encoding': self.file_encoding
        }

        try:
            for i, chunk in enumerate(pd.read_csv(self.input_file_path, **read_csv_args)):
                current_chunk_rows = len(chunk)
                print(f"  Pass 1: Processing chunk {i+1} (read {current_chunk_rows} valid rows)...", end='\r')

                if current_chunk_rows == 0:
                    continue

                # Initialize missing_counts and column_names on the first chunk
                if missing_counts is None:
                    missing_counts = pd.Series(0, index=chunk.columns)
                    column_names = chunk.columns.tolist()

                # Update missing value counts for the current chunk
                current_missing = chunk.isnull().sum()
                current_missing = current_missing.reindex(missing_counts.index, fill_value=0)
                missing_counts = missing_counts.add(current_missing, fill_value=0)
                self.total_rows_processed_pass1 += current_chunk_rows

            print("\nPass 1: Finished analyzing chunks.")
            if self.total_rows_processed_pass1 == 0:
                print("Error: No valid data processed in Pass 1.")
                sys.exit(1)

            # Calculate the percentage of missing values for each column
            missing_percentage = missing_counts / self.total_rows_processed_pass1
            # Retain columns with missing percentage below the threshold
            self.columns_to_keep = missing_percentage[missing_percentage < self.missing_threshold].index.tolist()

            print(f"Pass 1 Complete. Total valid rows analyzed: {self.total_rows_processed_pass1}")
            print(f"Columns to keep: {len(self.columns_to_keep)}")
            if not self.columns_to_keep:
                print("Error: No columns met the criteria to be kept.")
                sys.exit(1)

        except FileNotFoundError:
            print(f"Error: Input file not found at '{self.input_file_path}'")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: Input file '{self.input_file_path}' is empty.")
            sys.exit(1)

    def write_jsonl(self):
        # Second pass: Write the filtered data to a JSONL file
        print("\n--- Pass 2: Writing JSONL ---")
        read_csv_args = {
            'sep': self.delimiter,
            'chunksize': self.chunk_size,
            'low_memory': False,
            'on_bad_lines': self.handle_bad_lines,
            'encoding': self.file_encoding,
            'usecols': self.columns_to_keep  # Only read the columns identified in Pass 1
        }

        try:
            # Ensure the output file is empty before appending data
            with open(self.output_json_path, 'w', encoding=self.file_encoding) as f_out:
                pass

            for i, chunk in enumerate(pd.read_csv(self.input_file_path, **read_csv_args)):
                current_chunk_rows = len(chunk)
                print(f"  Pass 2: Processing chunk {i+1} ({current_chunk_rows} rows) -> JSONL...", end='\r')

                if current_chunk_rows > 0:
                    # Convert the chunk to JSONL format and append to the output file
                    json_string = chunk.to_json(orient='records', lines=True, force_ascii=False)
                    with open(self.output_json_path, 'a', encoding=self.file_encoding) as f_out:
                        f_out.write(json_string)
                    self.total_rows_processed_pass2 += current_chunk_rows

            print("\nPass 2: Finished writing chunks.")
            print(f"Successfully processed {self.total_rows_processed_pass2} rows.")

        except (IOError, OSError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    def process(self):
        # Main method to execute the data cleaning process
        self.validate_paths()  # Validate input and output paths
        self.identify_columns_to_keep()  # Identify columns to retain
        self.write_jsonl()  # Write the filtered data to JSONL format
