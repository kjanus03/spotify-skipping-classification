from astral import LocationInfo
from astral.sun import sun
from datetime import datetime
from dateutil import tz
from collections import defaultdict
from datetime import timedelta


class SunTimes:
    def __init__(self, location: tuple[float, float] = (51.1079, 17.0385)):
        lat, lon = location
        self.location = LocationInfo(latitude=lat, longitude=lon)
        self.tz = tz.gettz('Europe/Warsaw')
        self.cache = defaultdict(dict)
        self.times_descriptions = {
            'wschód słońca': '\nod 30 minut przed wschodem słońca\n do wschodu słońca',
            'rano': '\nod wschodu słońca\n do 4 godzin po wschodzie słońca',
            'środek dnia': '\nod 4 godzin po wschodzie słońca\n do 4 godzin przed zachodem słońca',
            'po południu': '\nod 4 godzin przed zachodem słońca\n do 1 godziny przed zachodem słońca',
            'zachód słońca': '\nod 1 godziny przed zachodem słońca\n do 30 minut po zachodzie słońca',
            'noc': '\nod 30 minut po zachodzie słońca\n do 30 minut przed wschodem słońca'
        }

    def get_sunrise_sunset(self, date: datetime) -> tuple[datetime, datetime]:
        date_only = date.date()  # Remove time component for caching
        if date_only not in self.cache:
            s = sun(self.location.observer, date=date_only)
            sunrise_local = s['sunrise'].astimezone(self.tz)
            sunset_local = s['sunset'].astimezone(self.tz)
            self.cache[date_only] = (sunrise_local, sunset_local)
        return self.cache[date_only]

    def get_time_of_day(self, date: datetime) -> str:
        sunrise, sunset = self.get_sunrise_sunset(date)
        dawn = sunrise - timedelta(minutes=30)
        morning_end = sunrise + timedelta(hours=4)
        midday_end = sunset - timedelta(hours=4)
        afternoon_end = sunset - timedelta(hours=1)
        golden_hour_end = sunset + timedelta(minutes=30)

        if dawn <= date < sunrise:
            return 'wschód słońca'
        elif sunrise <= date < morning_end:
            return 'rano'
        elif morning_end <= date < midday_end:
            return 'środek dnia'
        elif midday_end <= date < afternoon_end:
            return 'po południu'
        elif afternoon_end <= date < golden_hour_end:
            return 'zachód słońca'
        else:
            return 'noc'
