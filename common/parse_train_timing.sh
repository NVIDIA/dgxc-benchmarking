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

# Parse train_step_timing and TFLOPS_per_GPU from experiment log files and calculate mean and std dev for iterations 11-44
# Usage: ./parse_train_timing.sh [options] [experiments_directory]

set -e

# Constants
# Iterations are zero indexed
readonly MIN_ITERATION=11
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
    -h, --help         Show this help message

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
            if [[ ! "$OUTPUT_FORMAT" =~ ^(table|csv|json)$ ]]; then
                echo "Error: Invalid format '$OUTPUT_FORMAT'. Use: table, csv, or json" >&2
                exit 1
            fi
            shift
            ;;
        --full-names)
            SHOW_FULL_NAMES=true
            shift
            ;;
        -h|--help)
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
    if [[ "$SHOW_FULL_NAMES" == "true" ]]; then
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
            if [[ "$status" == "Success" ]]; then
                printf "%-90s %8s %13s %12s %19s %18s\n" "$display_name" "Success" "$time_mean" "$time_std_dev" "$tflops_mean" "$tflops_std_dev"
            elif [[ "$status" == "Failed" ]]; then
                if [[ -n "$max_iter" ]]; then
                    printf "%-90s %8s %30s\n" "$display_name" "Failed" "($max_iter iterations)"
                else
                    printf "%-90s %8s %13s %12s %19s %18s\n" "$display_name" "Failed" "-" "-" "-" "-"
                fi
            fi
            ;;
        csv)
            if [[ "$status" == "Success" ]]; then
                echo "$filename,Success,$time_mean,$time_std_dev,$tflops_mean,$tflops_std_dev,"
            elif [[ "$status" == "Failed" ]]; then
                if [[ -n "$max_iter" ]]; then
                    echo "$filename,Failed,,,,,,$max_iter"
                else
                    echo "$filename,Failed,,,,,,"
                fi
            fi
            ;;
    esac
}

# Function to output header
output_header() {
    case "$OUTPUT_FORMAT" in
        table)
            echo "Train Step Timing and TFLOPS Analysis (iterations $MIN_ITERATION-$MAX_ITERATION)"
            echo "================================================================================"
            printf "%-90s %8s %13s %12s %19s %18s\n" "Experiment" "Status" "Time Mean (s)" "Time Std (s)" "TFLOPS_per_GPU Mean" "TFLOPS_per_GPU Std"
            printf "%-90s %8s %13s %12s %19s %18s\n" "$(printf '%*s' 90 '' | tr ' ' '-')" "--------" "-------------" "------------" "-------------------" "------------------"
            ;;
        csv)
            echo "filename,status,time_mean_seconds,time_std_dev_seconds,tflops_per_gpu_mean,tflops_per_gpu_std_dev,max_iteration"
            ;;
        json)
            echo "{"
            echo "  \"analysis\": {"
            echo "    \"min_iteration\": $MIN_ITERATION,"
            echo "    \"max_iteration\": $MAX_ITERATION,"
            echo "    \"experiments_directory\": \"$EXPERIMENTS_DIR\""
            echo "  },"
            echo "  \"results\": ["
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
                echo "  Success rate: $(( files_processed * 100 / total_experiment_files ))%"
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
            echo "  \"summary\": {"
            echo "    \"success_experiments\": $files_processed,"
            echo "    \"failed_experiments\": $failed_count,"
            if [[ $total_experiment_files -gt 0 ]]; then
                echo "    \"success_rate\": $(( files_processed * 100 / total_experiment_files ))"
            else
                echo "    \"success_rate\": null"
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

# Count total files
total_files=$(echo "$out_files" | wc -l)
files_processed=0
incomplete_count=0
failed_early_count=0

# Store results for JSON formatting
declare -a json_results

output_header

while IFS= read -r file; do
    filename=$(basename "$file")
    
    # Skip nccltrace files - these are massive logs and aren't full runs.
    if [[ "$filename" == *"nccltrace"* ]]; then
        continue
    fi
    
    # Check if file contains any train_step_timing data at all
    has_timing_data=$(grep -q "train_step_timing in s:" "$file" 2>/dev/null && echo "yes" || echo "no")
    
    if [[ "$has_timing_data" == "yes" ]]; then
        # Extract timing and TFLOPS data and calculate mean and std dev in single awk pass
        result=$(grep "train_step_timing in s:" "$file" 2>/dev/null | \
            awk -v min_iter="$MIN_ITERATION" -v max_iter="$MAX_ITERATION" '
            /iteration [0-9]+\/49/ {
                match($0, /iteration ([0-9]+)\/49/, iter_arr)
                iteration = iter_arr[1]
                
                if (iteration >= min_iter && iteration <= max_iter) {
                    match($0, /train_step_timing in s: ([0-9]+\.?[0-9]*)/, timing_arr)
                    match($0, /TFLOPS_per_GPU: ([0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?)/, tflops_arr)
                    if (timing_arr[1] != "" && tflops_arr[1] != "") {
                        count++
                        time_values[count] = timing_arr[1]
                        # Convert scientific notation to regular number
                        tflops_val = tflops_arr[1]
                        if (match(tflops_val, /[eE]/)) {
                            # Handle scientific notation
                            tflops_val = sprintf("%.10f", tflops_val)
                        }
                        tflops_values[count] = tflops_val
                        time_sum += timing_arr[1]
                        tflops_sum += tflops_val
                        if (iteration > max_found) max_found = iteration
                    }
                }
            }
            END {
                if (count > 0) {
                    if (max_found < max_iter) {
                        print "INCOMPLETE:" max_found
                    } else {
                        time_mean = time_sum / count
                        tflops_mean = tflops_sum / count
                        
                        # Calculate standard deviations
                        time_sum_sq_diff = 0
                        tflops_sum_sq_diff = 0
                        for (i = 1; i <= count; i++) {
                            time_diff = time_values[i] - time_mean
                            time_sum_sq_diff += time_diff * time_diff
                            
                            tflops_diff = tflops_values[i] - tflops_mean
                            tflops_sum_sq_diff += tflops_diff * tflops_diff
                        }
                        time_std_dev = sqrt(time_sum_sq_diff / count)
                        tflops_std_dev = sqrt(tflops_sum_sq_diff / count)
                        
                        printf "COMPLETE:%.3f:%.3f:%.2f:%.2f", time_mean, time_std_dev, tflops_mean, tflops_std_dev
                    }
                }
            }')
        
        if [ -n "$result" ]; then
            if [[ "$result" == INCOMPLETE:* ]]; then
                max_found=${result#INCOMPLETE:}
                if [[ "$OUTPUT_FORMAT" == "json" ]]; then
                    json_results+=("{\"filename\": \"$filename\", \"status\": \"Failed\", \"max_iteration\": $max_found}")
                else
                    output_result "$filename" "Failed" "" "" "" "" "$max_found"
                fi
                incomplete_count=$((incomplete_count + 1))
            elif [[ "$result" == COMPLETE:* ]]; then
                # Parse mean and std dev from result
                stats=${result#COMPLETE:}
                time_mean=$(echo "$stats" | cut -d: -f1)
                time_std_dev=$(echo "$stats" | cut -d: -f2)
                tflops_mean=$(echo "$stats" | cut -d: -f3)
                tflops_std_dev=$(echo "$stats" | cut -d: -f4)
                if [[ "$OUTPUT_FORMAT" == "json" ]]; then
                    json_results+=("{\"filename\": \"$filename\", \"status\": \"Success\", \"time_mean\": $time_mean, \"time_std_dev\": $time_std_dev, \"tflops_per_gpu_mean\": $tflops_mean, \"tflops_per_gpu_std_dev\": $tflops_std_dev}")
                else
                    output_result "$filename" "Success" "$time_mean" "$time_std_dev" "$tflops_mean" "$tflops_std_dev" ""
                fi
                files_processed=$((files_processed + 1))
            fi
        fi
    else
        # Check if this looks like an experiment file that should have had timing data
        # Look for patterns that indicate this was meant to be an experiment log
        # Exclude sbatch files and other non-experiment logs
        if [[ ! "$filename" =~ ^sbatch_ ]] && [[ ! "$filename" =~ ^vboost ]] && \
           grep -q "iteration\|training\|model\|experiment" "$file" 2>/dev/null; then
            failed_early_count=$((failed_early_count + 1))
            if [[ "$OUTPUT_FORMAT" == "json" ]]; then
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
if [[ "$OUTPUT_FORMAT" == "json" ]]; then
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
    echo "Error: No valid complete train_step_timing and TFLOPS data found in any .out files" >&2
    exit 1
fi 