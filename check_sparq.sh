# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

set -e
set -o xtrace

make -j main tests/test-sparq

./tests/test-sparq

./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f32 -ctv f32
./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f32 -ctv f32 --sparq -k1 0 -k2 0
./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f32 -ctv f32 --sparq -k1 0 -k2 0 --sparq-default-layout
./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f32 -ctv f32 --sparq -k1 32 -k2 8
./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f32 -ctv f32 --sparq -k1 32 -k2 8 --sparq-default-layout

if [[ $1 == "--half" ]] ; then
    ./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f16 -ctv f16
    ./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f16 -ctv f16 --sparq -k1 0 -k2 0
    ./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f16 -ctv f16 --sparq -k1 0 -k2 0 --sparq-default-layout
    ./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f16 -ctv f16 --sparq -k1 32 -k2 8
    ./main -m ./models/llama-2-7b.Q8_0.gguf -p 'How tall is the Eiffel tower?' -n 16 -e --temp 0 -ctk f16 -ctv f16 --sparq -k1 32 -k2 8 --sparq-default-layout
fi

echo -e "\n\n##### All checks passed! #####\n"
