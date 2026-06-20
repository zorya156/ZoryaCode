from datetime import datetime, timedelta


def time_to_end_of_summer():
    now = datetime.now()
    year = now.year
    end_of_summer = datetime(year, 9, 1)
    if now >= end_of_summer:
        end_of_summer = datetime(year + 1, 9, 1)
    delta = end_of_summer - now
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return days, hours, minutes, seconds


def main():
    days, hours, minutes, seconds = time_to_end_of_summer()
    print(f"До конца лета осталось: {days} дней, {hours} часов, {minutes} минут, {seconds} секунд.")


if __name__ == "__main__":
    main()


input("Нажмите Enter для выхода...")