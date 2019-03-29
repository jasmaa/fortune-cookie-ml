"""Scrapes fortune cookie fortunes into json
"""

from selenium import webdriver
import json

quotes = {}

options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(chrome_options=options)
n = 0

# scrape fortunes
while True:
    print(f"http://www.fortunecookiemessage.com/archive.php?start={n}")
    driver.get(f"http://www.fortunecookiemessage.com/archive.php?start={n}")
    elems = driver.find_elements_by_xpath("//a[@href]")

    count = 0
    for elem in elems:
        url = elem.get_attribute("href")

        try:
            idx = url.index("http://www.fortunecookiemessage.com/cookie/")
        except ValueError:
            continue

        if idx == 0:
            res = "".join(url.split("http://www.fortunecookiemessage.com/cookie/"))
            res = res.split("-")
            data_id = int(res[0])
            data = " ".join(res[1:])
            
            quotes[data_id] = data
            count += 1

    if count == 0:
        break
    else:
        n += count

# dump to json
with open("fortunes.json", "w") as f:
    f.write(json.dumps(quotes, indent=4))
    f.close()

driver.close()
print("Done!")
