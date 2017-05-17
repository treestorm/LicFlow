# import datetime
import matplotlib.dates
import matplotlib.pyplot as plt
import datetime

data = [
        [1484611200.0, 844.4333],
        [1484524800.0, 783.3373],
        [1484438400.0, 774.194 ],
        [1484352000.0, 769.2299]
    ]

x = [datetime.datetime.fromtimestamp(element[0]) for element in data]
y = [element[1] for element in data]

plt.plot( x,  y,  ls="-",  c= "b",  linewidth  = 2 )
plt.xlabel("Dates")

time_formatter = matplotlib.dates.DateFormatter("%Y-%m-%d")
plt.axes().xaxis.set_major_formatter(time_formatter)
plt.axes().xaxis_date() # this is not actually necessary

plt.show()