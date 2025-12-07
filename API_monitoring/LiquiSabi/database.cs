using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace LiquiSabi.database;
public class LogDatabase
{
    private readonly string _connectionString;

    public LogDatabase(string dbPath)
    {
        _connectionString = $"Data Source={dbPath}";
        EnsureTableExists();
    }

    private void EnsureTableExists()
    {
        using var connection = new SqliteConnection(_connectionString);
        connection.Open();

        var cmd = connection.CreateCommand();
        cmd.CommandText =
        @"
            CREATE TABLE IF NOT EXISTS Logs (
                Timestamp TEXT NOT NULL,
                Coordinator TEXT NOT NULL,
                Data TEXT NOT NULL
            );
        ";
        cmd.ExecuteNonQuery();
    }

    public void InsertLog(string coordinator, String data)
    {
        string timestamp = DateTime.UtcNow.ToString("o");
        string jsonData = JsonSerializer.Serialize(data);

        using var connection = new SqliteConnection(_connectionString);
        connection.Open();

        var cmd = connection.CreateCommand();
        cmd.CommandText =
        @"
            INSERT INTO Logs (Timestamp, Coordinator, Data)
            VALUES ($timestamp, $coordinator, $data);
        ";
        cmd.Parameters.AddWithValue("$timestamp", timestamp);
        cmd.Parameters.AddWithValue("$coordinator", coordinator);
        cmd.Parameters.AddWithValue("$data", data);
        cmd.ExecuteNonQuery();
    }
}