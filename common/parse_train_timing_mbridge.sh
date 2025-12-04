#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Parse train_step_timing and TFLOPS_per_GPU from experiment log files and calculate mean and std dev for iterations 35-44
# Usage: ./parse_train_timing.sh [options] [experiments_directory]

set -e

# Constants
readonly MIN_ITERATION=35
readonly MAX_ITERATION=44

# Default values
EXPERIMENTS_DIR="experiments"
OUTPUT_FORMAT="table"
SHOW_FULL_NAMES=false

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [options] [experiments_directory]

Options:
    --format=FORMAT     Output format: table (default), csv, json
    --full-names        Show full filenames instead of shortened versions
    -h, --help          Show this help message

Arguments:
    experiments_directory    Directory containing .out files (default: experiments)

Examples:
    $0                                    # Use default table format
    $0 --format=csv experiments           # CSV output
    $0 --format=json --full-names         # JSON with full filenames
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --format=*)
            OUTPUT_FORMAT="${1#*=}"
            if [[ ! $OUTPUT_FORMAT =~ ^(table|csv|json)$ ]]; then
                echo "Error: Invalid format '$OUTPUT_FORMAT'. Use: table, csv, or json" >&2
                exit 1
            fi
            shift
            ;;
        --full-names)
            SHOW_FULL_NAMES=true
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            EXPERIMENTS_DIR="$1"
            shift
            ;;
    esac
done

# Function to shorten filename for display
shorten_filename() {
    local filename="$1"
    if [[ $SHOW_FULL_NAMES == "true" ]]; then
        echo "$filename"
    else
        # Drop everything before the first period and remove _0.out extension
        local shortened
        shortened=$(echo "$filename" | sed -E 's/^[^.]*\.//; s/_0\.out$//')
        echo "$shortened"
    fi
}

# Function to output results based on format
output_result() {
    local filename="$1"
    local status="$2"
    local time_mean="$3"
    local time_std_dev="$4"
    local tflops_mean="$5"
    local tflops_std_dev="$6"
    local max_iter="$7"

    local display_name
    display_name=$(shorten_filename "$filename")

    case "$OUTPUT_FORMAT" in
        table)
            if [[ $status == "Success" ]]; then
                printf "%-90s %8s %13s %12s %19s %18s\n" "$display_name" "Success" "$time_mean" "$time_std_dev" "$tflops_mean" "$tflops_std_dev"
            elif [[ $status == "Failed" ]]; then
                if [[ -n $max_iter ]]; then
                    printf "%-90s %8s %30s\n" "$display_name" "Failed" "($max_iter iterations)"
                else
                    printf "%-90s %8s %13s %12s %19s %18s\n" "$display_name" "Failed" "-" "-" "-" "-"
                fi
            fi
            ;;
        csv)
            if [[ $status == "Success" ]]; then
                echo "$filename,Success,$time_mean,$time_std_dev,$tflops_mean,$tflops_std_dev,"
            elif [[ $status == "Failed" ]]; then
                if [[ -n $max_iter ]]; then
                    echo "$filename,Failed,,,,,$max_iter"
                else
                    echo "$filename,Failed,,,,,"
                fi
            fi
            ;;
        json)
            # JSON entries are collected in json_results in the main loop
            if [[ $status == "Success" ]]; then
                json_results+=("{\"filename\": \"$filename\", \"status\": \"Success\", \"time_mean_ms\": $time_mean, \"time_std_ms\": $time_std_dev, \"tflops_mean\": ${tflops_mean:-null}, \"tflops_std\": ${tflops_std_dev:-null}}")
            else
                if [[ -n $max_iter ]]; then
                    json_results+=("{\"filename\": \"$filename\", \"status\": \"Failed\", \"max_iteration\": $max_iter}")
                else
                    json_results+=("{\"filename\": \"$filename\", \"status\": \"Failed\"}")
                fi
            fi
            ;;
    esac
}

# Function to output header
output_header() {
    case "$OUTPUT_FORMAT" in
        table)
            echo "Elapsed Time (ms) and TFLOPS/GPU Analysis (iterations $MIN_ITERATION-$MAX_ITERATION)"
            echo "================================================================================"
            printf "%-90s %8s %13s %12s %19s %18s\n" "Experiment" "Status" "Time Mean (ms)" "Time Std (ms)" "TFLOPS_per_GPU Mean" "TFLOPS_per_GPU Std"
            printf "%-90s %8s %13s %12s %19s %18s\n" "$(printf '%*s' 90 '' | tr ' ' '-')" "--------" "-------------" "------------" "-------------------" "------------------"
            ;;
        csv)
            echo "filename,status,time_mean_ms,time_std_dev_ms,tflops_per_gpu_mean,tflops_per_gpu_std_dev,max_iteration"
            ;;
        json)
            echo "{"
            echo '  "analysis": {'
            echo "    \"min_iteration\": $MIN_ITERATION,"
            echo "    \"max_iteration\": $MAX_ITERATION,"
            echo "    \"experiments_directory\": \"$EXPERIMENTS_DIR\""
            echo "  },"
            echo '  "results": ['
            ;;
    esac
}

# Function to output footer with summary
output_footer() {
    local files_processed="$1"
    local incomplete_count="$2"
    local failed_early_count="$3"
    local total_experiment_files="$4"

    local failed_count=$((incomplete_count + failed_early_count))

    case "$OUTPUT_FORMAT" in
        table)
            echo ""
            echo "Summary:"
            echo "  Success experiments: $files_processed"
            echo "  Failed experiments: $failed_count"
            if [[ $total_experiment_files -gt 0 ]]; then
                echo "  Success rate: $((files_processed * 100 / total_experiment_files))%"
            else
                echo "  Success rate: N/A"
            fi
            ;;
        csv)
            # CSV doesn't need footer for parsing, but we can add a comment
            echo "# Summary: $files_processed success, $failed_count failed, $total_experiment_files total"
            ;;
        json)
            # Remove trailing comma from last entry and close JSON
            echo "  ],"
            echo '  "summary": {'
            echo "    \"success_experiments\": $files_processed,"
            echo "    \"failed_experiments\": $failed_count,"
            if [[ $total_experiment_files -gt 0 ]]; then
                echo "    \"success_rate\": $((files_processed * 100 / total_experiment_files))"
            else
                echo '    "success_rate": null'
            fi
            echo "  }"
            echo "}"
            ;;
    esac
}

if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Error: Directory '$EXPERIMENTS_DIR' not found" >&2
    echo "Usage: $0 [experiments_directory]" >&2
    exit 1
fi

out_files=$(find "$EXPERIMENTS_DIR" -name "*.out" -type f)

if [ -z "$out_files" ]; then
    echo "Error: No .out files found in $EXPERIMENTS_DIR" >&2
    exit 1
fi

# Count progress
files_processed=0
incomplete_count=0
failed_early_count=0

# Store results for JSON formatting
declare -a json_results

output_header

while IFS= read -r file; do
    filename=$(basename "$file")

    # Skip various non-workload log files.
    case "$filename" in
        *nccltrace* | *prepare_squad_dataset_exp* | *import_ckpt_exp* | *nsys_analysis*)
            continue
            ;;
    esac

    # Check if file contains any of the new-format timing data
    has_timing_data=$(grep -q -E "elapsed time per iteration \(ms\):|TFLOP\/s\/GPU" "$file" 2> /dev/null && echo "yes" || echo "no")

    if [[ $has_timing_data == "yes" ]]; then
        # AWK now:
        #  - captures the numeric token immediately before "TFLOP/s/GPU" (handles scientific notation),
        #  - stores it into last_tflop,
        #  - when an iteration line with "elapsed time per iteration (ms)" is found within the iteration window,
        #    it pairs that elapsed time with the last_tflop (and then clears last_tflop so it isn't reused).
        result=$(awk -v min_iter="$MIN_ITERATION" -v max_iter="$MAX_ITERATION" '
            # capture the numeric token right before TFLOP/s/GPU (handles 1234.5 and scientific)
            /TFLOP\/s\/GPU/ {
                if (match($0, /([0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?)\s*TFLOP\/s\/GPU/, tf_arr)) {
                    last_tflop = tf_arr[1]
                }
            }

            # iteration line (handle variable spacing around the slash)
            /iteration[[:space:]]*[0-9]+[[:space:]]*\/[[:space:]]*[0-9]+/ {
                if (match($0, /iteration[[:space:]]*([0-9]+)/, iter_arr)) {
                    iteration = iter_arr[1] + 0
                    if (iteration >= min_iter && iteration <= max_iter) {
                        if (match($0, /elapsed time per iteration \(ms\):[[:space:]]*([0-9]+(\.[0-9]+)?)/, ms_arr)) {
                            count++
                            time_values[count] = ms_arr[1] + 0
                            time_sum += time_values[count]

                            if (last_tflop != "") {
                                tflops_count++
                                tflops_values[tflops_count] = last_tflop + 0
                                tflops_sum += tflops_values[tflops_count]
                            }

                            if (iteration > max_found) max_found = iteration
                            last_tflop = ""
                        }
                    }
                }
            }

            END {
                if (count > 0) {
                    if (max_found < max_iter) {
                        print "INCOMPLETE:" max_found
                    } else {
                        time_mean = time_sum / count
                        # sample stddev for times
                        time_var = 0
                        for (i = 1; i <= count; i++) time_var += (time_values[i] - time_mean)^2
                        time_std = (count > 1 ? sqrt(time_var / (count - 1)) : 0)

                        tflops_mean_str = "-"
                        tflops_std_str = "-"

                        if (tflops_count > 0) {
                            tflops_mean = tflops_sum / tflops_count
                            tflops_var = 0
                            for (i = 1; i <= tflops_count; i++) tflops_var += (tflops_values[i] - tflops_mean)^2
                            tflops_std = (tflops_count > 1 ? sqrt(tflops_var / (tflops_count - 1)) : 0)
                            tflops_mean_str = sprintf("%.2f", tflops_mean)
                            tflops_std_str = sprintf("%.2f", tflops_std)
                        }

                        # time_mean/time_std in ms (3 decimals), tflops formatted to 2 decimals above
                        printf "COMPLETE:%.3f:%.3f:%s:%s", time_mean, time_std, tflops_mean_str, tflops_std_str
                    }
                }
            }' "$file")

        if [ -n "$result" ]; then
            if [[ $result == INCOMPLETE:* ]]; then
                max_found=${result#INCOMPLETE:}
                if [[ $OUTPUT_FORMAT == "json" ]]; then
                    json_results+=("{\"filename\": \"$filename\", \"status\": \"Failed\", \"max_iteration\": $max_found}")
                else
                    output_result "$filename" "Failed" "" "" "" "" "$max_found"
                fi
                incomplete_count=$((incomplete_count + 1))
            elif [[ $result == COMPLETE:* ]]; then
                # Parse mean and std dev from result
                stats=${result#COMPLETE:}
                time_mean=$(echo "$stats" | cut -d: -f1)
                time_std_dev=$(echo "$stats" | cut -d: -f2)
                tflops_mean=$(echo "$stats" | cut -d: -f3)
                tflops_std_dev=$(echo "$stats" | cut -d: -f4)

                # Prepare JSON-friendly values (null instead of "-")
                if [[ $tflops_mean == "-" ]]; then tflops_mean_json="null"; else tflops_mean_json="$tflops_mean"; fi
                if [[ $tflops_std_dev == "-" ]]; then tflops_std_json="null"; else tflops_std_json="$tflops_std_dev"; fi

                if [[ $OUTPUT_FORMAT == "json" ]]; then
                    json_results+=("{\"filename\": \"$filename\", \"status\": \"Success\", \"time_mean_ms\": $time_mean, \"time_std_ms\": $time_std_dev, \"tflops_mean\": $tflops_mean_json, \"tflops_std\": $tflops_std_json}")
                else
                    output_result "$filename" "Success" "$time_mean" "$time_std_dev" "$tflops_mean" "$tflops_std_dev" ""
                fi
                files_processed=$((files_processed + 1))
            fi
        fi
    else
        # If this looks like an experiment log but without timing lines, mark as early-failed
        if [[ ! $filename =~ ^sbatch_ ]] && [[ ! $filename =~ ^vboost ]] \
            && grep -qE "iteration|training|model|experiment" "$file" 2> /dev/null; then
            failed_early_count=$((failed_early_count + 1))
            if [[ $OUTPUT_FORMAT == "json" ]]; then
                json_results+=("{\"filename\": \"$filename\", \"status\": \"Failed\"}")
            else
                output_result "$filename" "Failed" "" "" "" "" ""
            fi
        fi
    fi
done <<< "$out_files"

# Calculate total experiment files (complete + incomplete + failed early)
total_experiment_files=$((files_processed + incomplete_count + failed_early_count))

# Output JSON results without trailing comma
if [[ $OUTPUT_FORMAT == "json" ]]; then
    for i in "${!json_results[@]}"; do
        if [[ $i -eq $((${#json_results[@]} - 1)) ]]; then
            echo "    ${json_results[$i]}"
        else
            echo "    ${json_results[$i]},"
        fi
    done
fi

output_footer "$files_processed" "$incomplete_count" "$failed_early_count" "$total_experiment_files"

if [ $files_processed -eq 0 ]; then
    echo "Error: No valid complete elapsed-time and TFLOPS data found in any .out files" >&2
    exit 1
fi
