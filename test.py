
import tools
haichart = tools.haichart("mcgs")
haichart.from_csv("example_datasets/FlightDelayStatistics.csv")
haichart.learning_to_rank()
haichart.eh_view = haichart.output_list("list")

print(haichart.eh_view)