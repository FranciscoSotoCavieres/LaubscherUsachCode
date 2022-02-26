import numpy as np
from Constants import NumericConstants

class ColumnMaximumExtractionEngine:

    _days: np.ndarray[float]
    _maximum_extraction_tonnage: np.ndarray[float]
    last_day: float

    def __init__(self, days: np.ndarray[float], maximum_extraction_tonnage: np.ndarray[float], last_day: float):
        self._days = days
        self._maximum_extraction_tonnage = maximum_extraction_tonnage
        self.last_day = last_day
        
    
        

#         public static int IndexOf(double value, double[] array)
#         {
#             for (var i = 0; i < (array.Length - 1); i++)
#                 if (array[i] - Tolerance <= value && value <= array[i + 1] + Tolerance)
#                     return i;
#             return -1;
#         }


#   public class MaximumExtractionEngine
#     {
#         /// <summary>
#         /// Gets or sets the days.
#         /// </summary>
#         /// <value>
#         /// The days.
#         /// </value>
#         private double[] Days { get; set; }
#         /// <summary>
#         /// Gets or sets the maximum extraction.
#         /// </summary>
#         /// <value>
#         /// The maximum extraction.
#         /// </value>
#         private double[] MaximumExtraction { get; set; }

#         /// <summary>
#         /// Gets or sets the last day.
#         /// </summary>
#         /// <value>
#         /// The last day.
#         /// </value>
#         public double LastDay { get; private set; }

#         /// <summary>
#         /// Gets or sets the linear interpolator.
#         /// </summary>
#         /// <value>
#         /// The linear interpolator.
#         /// </value>
#         private LinearInterpolation LinearInterpolator { get; set; } = new LinearInterpolation();

#         /// <summary>
#         /// The tolerance
#         /// </summary>
#         public const double Tolerance = 0.01;

#         /// <summary>
#         /// Initializes a new instance of the <see cref="MaximumExtractionEngine"/> class.
#         /// </summary>
#         /// <param name="days">The days.</param>
#         /// <param name="maximumExtraction">The maximum extraction.</param>
#         /// <param name="lastDay">The last day.</param>
#         public MaximumExtractionEngine(double[] days, double[] maximumExtraction, double lastDay)
#         {
#             Days = days;
#             MaximumExtraction = maximumExtraction;
#             LastDay = lastDay;
#         }
#         /// <summary>
#         /// Gets the maximum extraction.
#         /// </summary>
#         /// <param name="daysOfExtraction">The days of extraction.</param>
#         /// <param name="initialDay">The initial day.</param>
#         /// <returns></returns>
#         public MaximumExtractionInformation GetMaximumExtraction(double daysOfExtraction, double initialDay)
#         {
#             var information = new MaximumExtractionInformation();
#             if (initialDay + daysOfExtraction > LastDay)
#             {
#                 information.OutOfRange = true;
#                 information.IfExtractedDepleted = true;
#                 information.DaysOfExtraction = LastDay - initialDay;
#                 information.MaximumTonnage = GetTonnage(LastDay) - GetTonnage(initialDay);
#             }
#             else
#             {

#                 var actualTonnage = GetTonnage(initialDay);
#                 var finalTonnage = GetTonnage(initialDay + daysOfExtraction);
#                 information.DaysOfExtraction = daysOfExtraction;
#                 information.IfExtractedDepleted = false;
#                 information.OutOfRange = false;
#                 information.MaximumTonnage = finalTonnage - actualTonnage;
#             }
#             return information;
#         }

#         /// <summary>
#         /// Gets the day.
#         /// </summary>
#         /// <param name="tonnage">The tonnage.</param>
#         /// <returns></returns>
#         public double GetDay(double tonnage)
#         {
#             int index = IndexOf(tonnage, MaximumExtraction);
#             IndexCheck(index);
#             LinearInterpolator.computeCoefficients(Days[index], MaximumExtraction[index],
#                                                    Days[index + 1], MaximumExtraction[index + 1]);
#             return LinearInterpolator.computeXValue(tonnage);
#         }

#         /// <summary>
#         /// Gets the tonnage.
#         /// </summary>
#         /// <param name="day">The day.</param>
#         /// <returns></returns>
#         public double GetTonnage(double day)
#         {
#             var index = IndexOf(day, Days);
#             IndexCheck(index);
#             LinearInterpolator.computeCoefficients(Days[index], MaximumExtraction[index],
#                                                    Days[index + 1], MaximumExtraction[index + 1]);
#             return LinearInterpolator.computeYValue(day);
#         }

#         /// <summary>
#         /// Indexes the check.
#         /// </summary>
#         /// <param name="index">The index.</param>
#         /// <exception cref="Exception">Index outside boundaries</exception>
#         private static void IndexCheck(int index)
#         {
#             if (index == -1)
#                 throw new Exception("Index outside boundaries");
#         }
#         /// <summary>
#         /// Indexes the of.
#         /// </summary>
#         /// <param name="value">The value.</param>
#         /// <param name="array">The array.</param>
#         /// <returns></returns>
#         public static int IndexOf(double value, double[] array)
#         {
#             for (var i = 0; i < (array.Length - 1); i++)
#                 if (array[i] - Tolerance <= value && value <= array[i + 1] + Tolerance)
#                     return i;
#             return -1;
#         }
#     }
