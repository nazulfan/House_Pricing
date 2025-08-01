import time
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

def main():
    """
    Fungsi utama untuk membuka browser, menavigasi semua halaman,
    dan mengumpulkan semua link properti.
    """
    # --- KONFIGURASI ---
    # Anda bisa mengubah URL ini ke kota atau area lain jika perlu
    START_URL = "https://www.rumah123.com/sewa/rumah/?page=895"
    OUTPUT_CSV = "daftar_link_rumah3.csv"
    MAX_PAGES = 1605  # Pengaman agar tidak terjadi loop tak terbatas

    # --- Inisialisasi WebDriver ---
    options = webdriver.ChromeOptions()
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("start-maximized")
    # Hapus komentar di bawah ini untuk menjalankan di background
    # options.add_argument('--headless')

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    try:
        driver.get(START_URL)
        print(f"Membuka halaman awal: {START_URL}")

        # Tangani cookie jika ada
        time.sleep(random.uniform(3, 5))
        try:
            cookie_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie_button.click()
            print("Banner cookie ditemukan dan ditutup.")
        except TimeoutException:
            print("Banner cookie tidak ditemukan, melanjutkan.")

        # --- Logika Paginasi Otomatis Sambil Mengumpulkan Link ---
        page_count = 1
        all_hrefs = []

        while page_count <= MAX_PAGES:
            print(f"\n--- Memproses Halaman ke-{page_count} ---")
            
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "card-featured__middle-section"))
                )
                print("Konten halaman berhasil dimuat.")
            except TimeoutException:
                print("Gagal memuat konten halaman. Menghentikan proses.")
                break

            elements = driver.find_elements(By.CLASS_NAME, "card-featured__middle-section")
            print(f"Menemukan {len(elements)} properti di halaman ini.")
            
            links_on_this_page = 0
            for element in elements:
                try:
                    link_element = element.find_element(By.TAG_NAME, 'a')
                    href = link_element.get_attribute('href')
                    if href and href not in all_hrefs:
                        all_hrefs.append(href)
                        links_on_this_page += 1
                except NoSuchElementException:
                    continue
            
            print(f"Berhasil menambahkan {len(all_hrefs)} link baru. Total link terkumpul: {len(all_hrefs)}")

            try:
                next_page_button = driver.find_element(By.CSS_SELECTOR, "a[rel='next']")
                parent_li = next_page_button.find_element(By.XPATH, "./parent::li")
                if "disabled" in parent_li.get_attribute("class"):
                    print("Tombol 'Next Page' tidak aktif. Halaman terakhir tercapai.")
                    break
                else:
                    print("Mengklik tombol 'Next Page'...")
                    driver.execute_script("arguments[0].click();", next_page_button)
                    page_count += 1
                    time.sleep(random.uniform(3, 5))
            except NoSuchElementException:
                print("Tombol 'Next Page' tidak ditemukan. Halaman terakhir tercapai.")
                break

    except Exception as e:
        print(f"Terjadi kesalahan fatal: {e}")
    finally:
        print(f"\n{'='*50}")
        print("Proses pengumpulan link selesai.")
        if all_hrefs:
            df = pd.DataFrame(all_hrefs, columns=['url'])
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"Total {len(all_hrefs)} link unik telah disimpan ke '{OUTPUT_CSV}'")
        else:
            print("Tidak ada link yang berhasil dikumpulkan.")
        print(f"{'='*50}")
        driver.quit()

if __name__ == "__main__":
    main()
