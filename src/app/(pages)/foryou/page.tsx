import { BusinessExplore } from "@/app/modules/business/BusinessExplore";
import Toolbar from "@/app/modules/Toolbar";
import { Business } from "@/types/BusinessTypes";
import { Metadata } from "next";
export const metadata: Metadata = {
    title: 'For You',
    description: '...',
}

export default function forYouPage() {
    const MOCK_DATA: Business[] = [
        {
            "category": "Cafe, Bakery, Flowershop, Locally Sourced, Women Owned, Community Oriented",
            "description": "A café, bakery, and flower shop combining fresh, local food with beautiful floral arrangements to foster community.",
            "id": "1",
            "image": "https://lh5.googleusercontent.com/p/AF1QipMgW8m1hweQVtNc7CXp4qUTj6NQYz2hI1ymIVGm=w408-h271-k-no",
            "location": "300 Occidental Ave S, Seattle, WA 98104",
            "name": "The London Plane",
            "price": "2",
            "review": "4.3"
          },
          {
            "category": "Brewery, Eco-Friendly, Locally Sourced, Sustainable, Family Owned",
            "description": "Craft beer brewed with a focus on sustainability and environmental stewardship.",
            "id": "2",
            "image": "https://lh5.googleusercontent.com/p/AF1QipNVOgISSnnLPj2ZTSI-kei8XLU62eBOXFEO79u2=w408-h408-k-no",
            "location": "1050 N 34th St, Seattle, WA 98103",
            "name": "Fremont Brewing",
            "price": "2",
            "review": "4.7"
          },
          {
            "category": "Cafe, Bakery, Locally Sourced, Sustainable, Community Oriented",
            "description": "To roast and serve ethically sourced, farm-direct coffee while supporting local artists and non-profits.",
            "id": "3",
            "image": "https://lh5.googleusercontent.com/p/AF1QipPbXQS0AgE8zXwrh3i8eD8u4dBhyib8y1plGwNm=w408-h306-k-no",
            "location": "1005 E Pike St, Seattle, WA 98122",
            "name": "Caffé Vita",
            "price": "1",
            "review": "4.4"
          },
          {
            "category": "Bakery, Locally Sourced, Women Owned, Family Owned",
            "description": "Freshly baked cookies made with locally sourced ingredients, creating joy in the community.",
            "id": "4",
            "image": "https://lh5.googleusercontent.com/p/AF1QipN-1jhpuIqmtQ0zfS-r35X5mn2Lm3kdKoyFmCs=w408-h306-k-no",
            "location": "2570 NE University Village St, Seattle, WA 98105",
            "name": "Hello Robin",
            "price": "1",
            "review": "4.6"
          },
          {
            "category": "Bookstore, Locally Sourced, Community Oriented, Women Owned",
            "description": "A community bookstore focused on cookbooks and fostering culinary curiosity.",
            "id": "5",
            "image": "https://lh5.googleusercontent.com/p/AF1QipOZPTTDoKnx4vUzgcZ9O8CVmJ46dZ9qT-m_V7M0=w408-h306-k-no",
            "location": "4252 Fremont Ave N, Seattle, WA 98103",
            "name": "Book Larder",
            "price": "2",
            "review": "4.9"
          },
          {
            "category": "Seafood, Eco-Friendly, Locally Sourced, Sustainable, Community-Oriented",
            "description": "Source sustainable seafood and engage the community through joyful, hands-on experiences.",
            "id": "6",
            "image": "https://lh5.googleusercontent.com/p/AF1QipP-LUCoGwSSYMV_y_ZAGI_m427i8zjTMrmgrkWw=w424-h240-k-no",
            "location": "86 Pike Pl, Seattle, WA 98101",
            "name": "Pike Place Fish Market",
            "price": "3",
            "review": "4.6"
          },
          {
            "category": "Thai, Family Owned, Minority Owned, Community Oriented",
            "description": "A Thai restaurant providing traditional flavors with a modern twist, celebrating Thai culture in Seattle.",
            "id": "7",
            "image": "https://lh5.googleusercontent.com/p/AF1QipNO59kJvaPi8x0mCeaaB_PbmLTiP0DRZqNBy3TR=w426-h240-k-no",
            "location": "414 E Pine St, Seattle, WA 98122",
            "name": "Sugar Hill",
            "price": "2",
            "review": "4.2"
          },
          {
            "category": "Italian, Family Owned, Handcrafted, Locally Sourced",
            "description": "Crafting cured meats using traditional Italian methods and supporting local farmers.",
            "id": "8",
            "image": "https://lh5.googleusercontent.com/p/AF1QipMMTcxOE8TN3wRwF-3s0Wi7nwAGWgXMLTQJjPev=w408-h306-k-no",
            "location": "404 Occidental Ave S, Seattle, WA 98104",
            "name": "Salumi Artisan Cured Meats",
            "price": "1",
            "review": "4.7"
          },
          {
            "category": "Brewery, Sustainable, Locally Sourced, Eco-Friendly",
            "description": "Crafting dry ciders with a focus on sustainability and local ingredients.",
            "id": "9",
            "image": "https://lh5.googleusercontent.com/p/AF1QipNor3CZgvGAtweZfN36NTEc0a_C73yZ4UVjHVbK=w408-h272-k-no",
            "location": "4660 Ohio Ave S, Seattle, WA 98134",
            "name": "Seattle Cider Company",
            "price": "1",
            "review": "4.7"
          },
          {
            "category": "Brewery, Handcrafted, Women Owned, Locally Sourced",
            "description": "To provide refreshing, handcrafted ginger beer made with real, locally sourced ingredients.",
            "id": "10",
            "image": "https://lh5.googleusercontent.com/p/AF1QipO0zrAwO5MlX2JjXCfS0sbTQxwDkYiSGCrAwDfP=w408-h280-k-no",
            "location": "4626 26th Ave NE, Seattle, WA 98105",
            "name": "Rachel's Ginger Beer",
            "price": "1",
            "review": "4.7"
          },
          {
            "category": "Sweets, Eco-Friendly, Sustainable, Handcrafted, Locally Sourced",
            "description": "To create sustainable, fair-trade chocolate while promoting social and environmental responsibility.",
            "id": "11",
            "image": "https://lh5.googleusercontent.com/p/AF1QipPlNhlZUucGBWwE1ZmltGliym0QUNur2zes7Fg7=w480-h240-k-no",
            "location": "3400 Phinney Ave N, Seattle, WA 98103",
            "name": "Theo Chocolate",
            "price": "2",
            "review": "4.7"
          },
          {
            "category": "Brewery, Sustainable, Locally Sourced, Eco-Friendly",
            "description": "Brew 100% organic beer while minimizing environmental impact.",
            "id": "12",
            "image": "https://lh5.googleusercontent.com/p/AF1QipN_uZqCYC88L4Pz9APfFn79xMaRvcxgxtJDMXX0=w408-h272-k-no",
            "location": "1330 N Forest St, Bellingham, WA 98225",
            "name": "Aslan Brewing Company",
            "price": "2",
            "review": "4.6"
          },
          {
            "category": "Bakery, Locally Sourced, Women Owned, Family Owned",
            "description": "A bakery offering cupcakes and ice cream made with local, natural ingredients, advocating for community causes.",
            "id": "13",
            "image": "https://streetviewpixels-pa.googleapis.com/v1/thumbnail?panoid=DSckZTWQcBYq8n-pgFvWsA&cb_client=search.gws-prod.gps&w=408&h=240&yaw=280&pitch=0&thumbfov=100",
            "location": "1101 34th Ave, Seattle, WA 98122",
            "name": "Cupcake Royale",
            "price": "1",
            "review": "4.5"
          },
          {
            "category": "Vegan, Minority Owned, Women Owned, Eco-Friendly",
            "description": "A vegan restaurant serving innovative plant-based dishes to inspire conscious and sustainable eating.",
            "id": "14",
            "image": "https://lh5.googleusercontent.com/p/AF1QipNs8eyCuMlcyohOexr9ov9q9guqzRAg7TxZCrtV=w480-h360-k-no",
            "location": "1124 Pike St, Seattle, WA 98101",
            "name": "Plum Bistro",
            "price": "3",
            "review": "4.4"
          },
          {
            "category": "Seafood, Sustainable, Locally Sourced, Eco-Friendly",
            "description": "To serve responsibly sourced oysters and seafood while fostering a culture of sustainability.",
            "id": "15",
            "image": "https://lh5.googleusercontent.com/p/AF1QipMWdqTwEkDTxVnepvlX8-yqfLTwVhfKPvIoyTiJ=w426-h240-k-no",
            "location": "86 Pine St, Seattle, WA 98101",
            "name": "The Walrus and the Carpenter",
            "price": "4",
            "review": "4.7"
          },
          {
            "category": "Cafe, Minority Owned, Community Oriented, Locally Sourced",
            "description": "Bring African coffee to the Pacific Northwest and create community connections",
            "id": "16",
            "image": "https://lh5.googleusercontent.com/p/AF1QipNERlxwFaRKztkMz8MaJli_pvn1rtjh46e9_tFs=w408-h306-k-no",
            "location": "4326 University Wy NE, Seattle, WA 98105",
            "name": "Boon Boona Coffee",
            "price": "1",
            "review": "4.5"
          },
          {
            "category": "Farm to Table, Sustainable, Eco-Friendly, Locally Sourced",
            "description": "Promote sustainable food systems through fresh, locally sourced sandwiches and salads.",
            "id": "17",
            "image": "https://lh5.googleusercontent.com/p/AF1QipMRuxxZZEsi1Kr-JsV-hmZd_I2qHXYWUoSJAYp6=w426-h240-k-no",
            "location": "999 3rd Ave Corner of 2nd &, Marion St, Seattle, WA 98104",
            "name": "Homegrown",
            "price": "1",
            "review": "4"
          },
          {
            "category": "Pizza, Family Owned, Locally Sourced, Community Oriented",
            "description": "Serve New York-style pizza using local ingredients in a family-friendly environment.",
            "id": "18",
            "image": "https://lh5.googleusercontent.com/p/AF1QipNpwjpuyN6WqiFv1CdK_93iW_-5RWT1NUFjYR9u=w408-h306-k-no",
            "location": "5107 Ballard Ave NW, Seattle, WA 98107",
            "name": "Ballard Pizza Company",
            "price": "1",
            "review": "4.2"
          },
          {
            "category": "Makerspace, Women Owned, Community Oriented, Handcrafted",
            "description": "A DIY school offering workshops on cooking and crafting, promoting hands-on learning and creativity.",
            "id": "19",
            "image": "https://lh5.googleusercontent.com/p/AF1QipM_H-nUaW5L5wVAJnVLleLFAOK6pPHOe1yv5vsq=w408-h543-k-no",
            "location": "3516 Fremont Pl N, Seattle, WA 98103",
            "name": "The Works Seattle",
            "price": "2",
            "review": "5"
          },
          {
            "category": "Plants, Clothing, Handcrafted, Locally Sourced, Community Oriented",
            "description": "A retail and plant shop focusing on thoughtful products and greenery, fostering meaningful community interactions.",
            "id": "20",
            "image": "https://lh5.googleusercontent.com/p/AF1QipN-mGi0TxenIBJnHyitEtZeNhwbpfIyktXms0gm=w408-h612-k-no",
            "location": "1525 Melrose Ave, Seattle, WA 98122",
            "name": "Glasswing",
            "price": "2",
            "review": "4.5"
          }
    ];

    return (
        <div className="bg-[#f4f4f4]">
            <div className="flex justify-center items-center min-h-screen">
                <BusinessExplore businesses={MOCK_DATA} />
            </div>
        </div>
    );
}