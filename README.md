
## Example: ทำบอท ETH/USDT:

- ก๊อปปี้ downloadData.yaml แล้วแก้ชื่อเป็น downloadData_eth.yaml ข้างในใส่ asset: ETHUSDT

- รันโหลดข้อมูล: python scripts/00_download_data.py --config downloadData_eth.yaml

- รันสร้างฟีเจอร์: python scripts/01_build_dataset.py --config downloadData_eth.yaml

- รันหั่นไฟล์: python scripts/01b_prepare_chunks.py --config downloadData_eth.yaml

- รันเทรนโมเดล: python scripts/04_train_chunked.py --pair eth