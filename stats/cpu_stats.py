## get utils

file_name = "CPU_Stats.csv"

write_data(file_name, [], ["CPU_Count", "CPU_Count (actual)", "CPU_Stats", "CPU_Load_Avg", "CPU_Frequency"])

write_data(file_name, [psutil.cpu_count(), psutil.cpu_count(logical=False), psutil.cpu_stats(), psutil.getloadavg(), psutil.cpu_freq(percpu=True)])

file_name = "RAM_Stats.csv"
write_data(file_name, [], ["Virtual_Memory", "Swap Memory"])
write_data(file_name, [psutil.virtual_memory(), psutil.swap_memory()])