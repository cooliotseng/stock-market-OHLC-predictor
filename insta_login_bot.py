from time import sleep

from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait


class InstaBot:
    def __init__(self):
        self.driver = webdriver.Chrome(executable_path='/usr/lib/chromium-browser/chromedriver')
        self.driver.get('https://www.instagram.com/')
        sleep(2)
        self.driver.find_element_by_xpath("//input[@name=\"username\"]").send_keys("siddheshs3000")
        self.driver.find_element_by_xpath("//input[@name=\"password\"]").send_keys("cjhATY53")
        self.driver.find_element_by_xpath('//button[@type="submit"]').click()
        # sleep(10)
        notNowButton = WebDriverWait(self.driver, 15).until(
        lambda d: d.find_element_by_xpath('//button[text()="Not Now"]')
        )
        notNowButton .click()
        NotNowButton = WebDriverWait(self.driver, 15).until(
        lambda d: d.find_element_by_xpath('//button[text()="Not Now"]')
        )
        NotNowButton .click()
        sleep(10)









InstaBot()
