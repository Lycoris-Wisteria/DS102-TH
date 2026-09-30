import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

app_id = os.getenv("app_id")
app_key = os.getenv("app_key")



def get_jobs(keyword, location, country, num_pages=20, results_per_page=50):
    all_jobs = []

    for page in range(1, num_pages + 1):

        print(f"Đang lấy page {page}/{num_pages}...")

        page_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

        params = {
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": results_per_page,
            "what": keyword,
            "where": location,
            "content-type": "application/json"
        }

        response = requests.get(page_url, params=params)

        print("Status:", response.status_code)

        if response.status_code != 200:
            print(response.text[:500])
            continue

        data = response.json()

        jobs = data.get("results", [])

        print("Số job thu được:", len(jobs))

        all_jobs.extend(jobs)

        time.sleep(1)

    return all_jobs

def create_dataframe(all_jobs):
    df = pd.DataFrame(all_jobs)

    # Company
    if "company" in df.columns:
        df["company_name"] = df["company"].apply(
            lambda x: x.get("display_name")
            if isinstance(x, dict)
            else None
        )

    # Location
    if "location" in df.columns:
        df["location_name"] = df["location"].apply(
            lambda x: x.get("display_name")
            if isinstance(x, dict)
            else None
        )

    # Category
    if "category" in df.columns:
        df["category_name"] = df["category"].apply(
            lambda x: x.get("label")
            if isinstance(x, dict)
            else None
        )

    columns = [
        "id",
        "title",
        "company_name",
        "location_name",
        "latitude",
        "longitude",
        "search_country",
        "search_location",
        "search_keyword",
        "category_name",
        "salary_min",
        "salary_max",
        "salary_is_predicted",
        "contract_type",
        "contract_time",
        "created",
        "description",
        "redirect_url",
    ]

    columns = [
        col for col in columns
        if col in df.columns
    ]
    
    df = df[columns]

    # Xóa job trùng
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")

    return df

def save_csv(df, filename="adzuna_jobs.csv"):
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"Đã lưu {len(df)} jobs vào {filename}")


def main():
    jobs = [
        # Anh(UK)
        ("Software engineer", "London", "gb"),
        ("Data analyst", "London", "gb"),
        ("Data science", "London", "gb"),
        ("DevOps engineer", "London", "gb"),
        ("Web developer", "London", "gb"),

        # Mỹ(U)S
        ("Software engineer", "New York", "us"),
        ("Data analyst", "New York", "us"),
        ("Data science", "New York", "us"),
        ("DevOps engineer", "New York", "us"),
        ("Web developer", "New York", "us")  
    ]

    all_jobs = []

    for keyword, location, country in jobs:

        print(f"\nĐang lấy nghề: {keyword}-{location}-{country}")

        data = get_jobs(
            keyword=keyword,
            location=location,
            country=country,
            num_pages=10,
            results_per_page=50
        )

        for job in data:
            job["search_country"] = country
            job["search_location"] = location
            job["search_keyword"] = keyword

        all_jobs.extend(data)

    df=create_dataframe(all_jobs)
    print("Tổng số job thu được: ", len(df))
    save_csv(df)

if __name__ == "__main__":
    main()

