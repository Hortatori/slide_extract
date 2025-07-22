N_SEQUENCES=10

echo "Sample $N_SEQUENCES sequences per channel..."

xan groupby channel,sequence_id "sum(duration) as duration" result_test_sequence_id.csv | \
xan sample --groupby channel $N_SEQUENCES --seed 25 | \
xan search --patterns - --pattern-column sequence_id sequence_id result_test_sequence_id.csv -e \
> sampled_sequences.csv

echo "Done."