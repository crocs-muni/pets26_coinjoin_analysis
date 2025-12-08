using LiquiSabi.ApplicationCore.Utils.Tor.Http;
using LiquiSabi.ApplicationCore.Utils.WabiSabi.Client;
using LiquiSabi.ApplicationCore.Utils.WabiSabi.Models;
using LiquiSabi.ApplicationCore.Publishing.Nostr;
using Newtonsoft.Json;
using LiquiSabi.database;
using LiquiSabi.ApplicationCore.Utils.WabiSabi.Models.Serialization;

namespace LiquiSabi;

record ApiClientWithCoordinator(
    CoordinatorDiscovery.Coordinator Coordinator,
    WabiSabiHttpApiClient ApiClient);

public static class Program
{

    public static async Task Main()
    {
        while (true)
        {
            var hdb = new LogDatabase("human_db.sqlite");
            var sdb = new LogDatabase("status_db.sqlite");

            var urls = File.ReadLines("urls.txt");
            foreach (var url in urls)
            {
                try
                {
                    var httpClient = new HttpClient();
                    httpClient.BaseAddress = new Uri(url);
                    var clearnetHttpClient = new ClearnetHttpClient(httpClient);
                    var apiClient = new WabiSabiHttpApiClient(clearnetHttpClient);
                    using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(60));
                    using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cts.Token, new CancellationToken());
                    var status = await apiClient.GetStatusAsync(RoundStateRequest.Empty, linkedCts.Token);
                    var humanMonitor = await apiClient.GetHumanMonitor(new HumanMonitorRequest(), linkedCts.Token);

                    hdb.InsertLog(url, JsonConvert.SerializeObject(humanMonitor, JsonSerializationOptions.Default.Settings));
                    foreach (var state in status.RoundStates)
                    {
                        sdb.InsertLog(url, JsonConvert.SerializeObject(state, JsonSerializationOptions.Default.Settings));
                    }
                    Console.WriteLine("Success for " + url);

                }
                catch (Exception e) { Console.WriteLine("Failed for " + url); Console.WriteLine(e); }
            }
            Thread.Sleep(60000);
        }
    }
}