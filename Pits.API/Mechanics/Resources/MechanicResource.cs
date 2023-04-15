
namespace Mecanillama.API.Mechanics.Resources;

public class MechanicResource
{
    public long Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Address { get; set; } = "undefined";
    public string Description { get; set; } = "undefined";
    public long Phone { get; set; }
    public long UserId { get; set; }
    
}