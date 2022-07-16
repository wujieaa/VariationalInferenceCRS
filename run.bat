
::TIMEOUT /T 5 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae2_result_m1 --model_name CEBVAE --epoch 100
::TIMEOUT /T 300 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae2_result_m2 --model_name CEBVAE2 --epoch 100
::TIMEOUT /T 300 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae2_result_m3 --model_name CEBVAE3 --epoch 100
::TIMEOUT /T 300 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae2_result_m4 --model_name CEBVAE4 --epoch 100
::TIMEOUT /T 300 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae2_result_m5 --model_name CEBVAE5 --epoch 100

::python -m model.my_model3.main --result_file_name bvae3_result_m1 --model_name CEBVAE --epoch 200
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae3_result_m2 --model_name CEBVAE2 --epoch 200
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae3_result_m3 --model_name CEBVAE3 --epoch 200
::TIMEOUT /T 900 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae3_result_m4 --model_name CEBVAE4 --epoch 200
::TIMEOUT /T 900 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae3_result_m5 --model_name CEBVAE5 --epoch 200
::TIMEOUT /T 900 /NOBREAK
::python -m model.my_model3.main --result_file_name bvae3_result_m6 --model_name CEBVAE6 --epoch 200
::TIMEOUT /T 900 /NOBREAK

python -m model.my_model4.main --result_file_name bvae4_result_m1 --model_name CEBVAE --epoch 300
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m2 --model_name CEBVAE2 --epoch 200
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m3 --model_name CEBVAE3 --epoch 200
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m4 --model_name CEBVAE4 --epoch 200
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m5 --model_name CEBVAE5 --epoch 200
TIMEOUT /T 1200 /NOBREAK
python -m model.my_model4.main --result_file_name bvae4_result_m6 --model_name CEBVAE6 --epoch 300

::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m1_001 --model_name CEBVAE --epoch 200 --keyphrase_weight 0.01
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m2_001 --model_name CEBVAE2 --epoch 200 --keyphrase_weight 0.01
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m3_001 --model_name CEBVAE3 --epoch 200 --keyphrase_weight 0.01
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m4_001 --model_name CEBVAE4 --epoch 200 --keyphrase_weight 0.01
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m5_001 --model_name CEBVAE5 --epoch 200 --keyphrase_weight 0.01
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m6_001 --model_name CEBVAE6 --epoch 200 --keyphrase_weight 0.01

::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m1_zo --model_name CEBVAE --epoch 200 --fusion_type zero_out
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m2_zo --model_name CEBVAE2 --epoch 200 --fusion_type zero_out
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m3_zo --model_name CEBVAE3 --epoch 200 --fusion_type zero_out
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m4_zo --model_name CEBVAE4 --epoch 200 --fusion_type zero_out
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m5_zo --model_name CEBVAE5 --epoch 200 --fusion_type zero_out
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model4.main --result_file_name bvae4_result_m6_zo --model_name CEBVAE6 --epoch 200 --fusion_type zero_out

::python -m model.my_model5.main --result_file_name bvae5_result --model_name CEBVAE --epoch 200
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model5.main --result_file_name bvae5_result_001 --model_name CEBVAE --epoch 200 --keyphrase_weight 0.01
::TIMEOUT /T 1200 /NOBREAK
::python -m model.my_model5.main --result_file_name bvae5_result_zo --model_name CEBVAE --epoch 200 --fusion_type zero_out
