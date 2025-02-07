import scripting
import argparse

'''
- output type: fasta, aln, tsv (flags -f, -a, -t)
    - determine whether to call the directory function based on the number of inputs
- print flag: -p 
- path input > gets processed in the scripting (there's a function)
- path output (optional) - change the name of the output single file

main scripting function(input, output_type = f/a/t, print = True/False, output = "")
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate embedding sequences and identify Leucine-Rich Repeats.")

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("-f", "--fasta", action="store_const", const="f", help="Output in FASTA format")
    output_group.add_argument("-a", "--aln", action="store_const", const="a", help="Output in ALN format")
    output_group.add_argument("-t", "--tsv", action="store_const", const="t", help="Output in TSV format")

    parser.add_argument("input_path", type=str, help="Path to the input file or directory")
    parser.add_argument("--hdf", type=str, default="20241205_hTLR_pool.hdf", help="Path to the HDF file used for clustering mapping (default: 20241205_hTLR_pool.hdf)")
    parser.add_argument("--output", type=str, default="", help="Custom output file name (default: auto-generated)")

    parser.add_argument("-p", "--print_flag", action="store_true", help="Print output to console")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Determine the output type from mutually exclusive group
    output_type = args.fasta or args.aln or args.tsv  # Will be "f", "a", or "t"

    # Call the main scripting function
    scripting.run_pipeline(input_path=args.input_path, hdf_file=args.hdf, 
                           output_type=output_type, print_flag=args.print_flag, 
                           output_path=args.output)

if __name__ == "__main__":
    main()
