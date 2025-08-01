import time
import re
import pandas as pd
import random
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# --- FUNGSI BANTUAN ---
def konversi_harga(harga_str):
    if not isinstance(harga_str, str): return None
    angka_str = re.search(r'[\d.]+', harga_str)
    if not angka_str: return None
    angka = float(angka_str.group().replace('.', ''))
    if 'juta' in harga_str.lower(): angka *= 1_000_000
    elif 'miliar' in harga_str.lower(): angka *= 1_000_000_000
    if '/bulan' in harga_str.lower(): angka *= 12
    return int(angka)

def clean_text(text):
    return text.replace('\n', ' ').strip() if text else None

# --- FUNGSI UNTUK SCRAPE HALAMAN DETAIL ---
def scrape_detail_page_selenium(driver):
    details = {}
    details['URL'] = driver.current_url # Penting untuk checkpoint
    try:
        details['Harga Sewa'] = konversi_harga(driver.find_element(By.CSS_SELECTOR, "span.text-primary.font-bold").text)
    except NoSuchElementException:
        details['Harga Sewa'] = None
    try:
        details['Lokasi'] = clean_text(driver.find_element(By.XPATH, "//h1/following-sibling::p").text)
    except NoSuchElementException:
        details['Lokasi'] = "N/A"
    try:
        wait = WebDriverWait(driver, 5)
        lihat_semua_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Muat lebih banyak')]")))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", lihat_semua_button)
        time.sleep(random.uniform(0.5, 1.0))
        lihat_semua_button.click()
        time.sleep(random.uniform(1.0, 1.5))
    except (TimeoutException, NoSuchElementException):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(random.uniform(1.0, 1.5))
    
    fitur_list = {
        'Kamar Tidur': "//p[contains(text(), 'Tidur') and not(contains(text(), 'Pembantu'))]/following-sibling::p",
        'Kamar Mandi': "//p[contains(text(), 'Mandi') and not(contains(text(), 'Pembantu'))]/following-sibling::p",
        'Luas Tanah': "//p[contains(text(), 'Luas Tanah')]/following-sibling::p",
        'Luas Bangunan': "//p[contains(text(), 'Luas Bangunan')]/following-sibling::p",
        'Carport': "//p[contains(text(), 'Carport')]/following-sibling::p",
        'Sertifikat': "//p[contains(text(), 'Sertifikat')]/following-sibling::p",
        'Daya Listrik': "//p[contains(text(), 'Daya Listrik')]/following-sibling::p",
        'Kamar Tidur Pembantu': "//p[contains(text(), 'Tidur Pembantu')]/following-sibling::p",
        'Kamar Mandi Pembantu': "//p[contains(text(), 'Mandi Pembantu')]/following-sibling::p",
        'Garasi': "//p[contains(text(), 'Garasi')]/following-sibling::p",
        'Jumlah Lantai': "//p[contains(text(), 'Jumlah Lantai')]/following-sibling::p",
        'Kondisi Properti': "//p[contains(text(), 'Kondisi Properti')]/following-sibling::p"
    }

    for nama_fitur, xpath in fitur_list.items():
        try:
            value = driver.find_element(By.XPATH, xpath).text
            angka = re.search(r'\d+', value)
            if nama_fitur in ['Sertifikat', 'Kondisi Properti']:
                details[nama_fitur] = value
            elif angka:
                details[nama_fitur] = int(angka.group())
            else:
                details[nama_fitur] = value
        except (NoSuchElementException, AttributeError):
            if nama_fitur in ['Carport', 'Garasi', 'Kamar Tidur Pembantu', 'Kamar Mandi Pembantu']:
                details[nama_fitur] = 0
            elif nama_fitur in ['Sertifikat', 'Kondisi Properti']:
                 details[nama_fitur] = "Lainnya"
            else:
                details[nama_fitur] = None
    return details

# --- FUNGSI UTAMA ---
def main():
    # --- KONFIGURASI ---
    INPUT_CSV = "daftar_link_rumah2.csv"
    OUTPUT_CSV = "database_sewa_rumah_final3.csv"

    # --- OPSI SELENIUM ---
    options = webdriver.ChromeOptions()
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("start-maximized")
    # options.add_argument('--headless')

    # --- LOGIKA CHECKPOINT ---
    try:
        df_links = pd.read_csv(INPUT_CSV)
        all_links_to_scrape = df_links['url'].tolist()[3000:3400]
    except FileNotFoundError:
        print(f"Error: File '{INPUT_CSV}' tidak ditemukan. Jalankan script 'collect_links.py' terlebih dahulu.")
        returna

    scraped_links = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_scraped = pd.read_csv(OUTPUT_CSV)
            if 'URL' in df_scraped.columns:
                scraped_links = set(df_scraped['URL'])
                print(f"Ditemukan {len(scraped_links)} link yang sudah di-scrape sebelumnya. Melanjutkan proses...")
        except pd.errors.EmptyDataError:
            print(f"File '{OUTPUT_CSV}' kosong. Memulai dari awal.")
        except Exception as e:
            print(f"Gagal membaca file output: {e}. Memulai dari awal.")
    
    links_to_process = [link for link in all_links_to_scrape if link not in scraped_links]
    
    if not links_to_process:
        print("Semua link sudah di-scrape. Proses selesai.")
        return
        
    print(f"Total link yang akan diproses: {len(links_to_process)} dari {len(all_links_to_scrape)}")
    
    # --- LOOP UTAMA: KUNJUNGI SETIAP LINK YANG TERSISA ---
    for i, url in enumerate(links_to_process):
        driver = None
        try:
            print(f"\n--- Memproses properti {i+1}/{len(links_to_process)} (Link Asli #{all_links_to_scrape.index(url)+1}) ---")
            print(f"URL: {url}")

            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            driver.get(url)
            
            if "Just a moment..." in driver.title or "Verifying you are human" in driver.title:
                print("     -> Properti dilindungi CAPTCHA. Melewati...")
                continue

            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.text-primary.font-bold")))
            
            details = scrape_detail_page_selenium(driver)
            
            if details and details.get('Harga Sewa'):
                # --- PERBAIKAN UTAMA: Simpan langsung ke file ---
                df_single_row = pd.DataFrame([details])
                # Tulis header jika file belum ada, jika sudah ada, tambahkan baris baru tanpa header
                df_single_row.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
                print("     -> Data berhasil di-scrape dan disimpan.")
            else:
                print("     -> Gagal mendapatkan data harga, properti dilewati.")
            
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            print(f"     -> Terjadi error. Melanjutkan... Error: {e}")
            continue
        finally:
            if driver:
                driver.quit()

    print(f"\n{'='*50}")
    print("Semua proses scraping detail selesai.")

if __name__ == "__main__":
    main()